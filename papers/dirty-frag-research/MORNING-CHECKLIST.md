# Dirty Frag — Phase 3/4 verification checklist

> Builds completed overnight 2026-05-08 ~02:25 EDT.  You are now at the
> qemu-verify gate.  Estimated remaining time: 60-90 min including the
> email send at the end.

## Fast path — `dirtyfrag` driver script (recommended)

`papers/dirty-frag-research/scripts/dirtyfrag` is a single bash driver
that wraps every Pi-side command in this checklist with auto-detection
of which side you are on (Mac vs Pi) and idempotent prep.  Mirrored to
the Pi at `/home/perseus/dirtyfrag/dirtyfrag` so the same verbs work
from either side.

```bash
# from anywhere (Mac or Pi):
dirtyfrag status                # live phase tracker — what's done, what's left
dirtyfrag verify                # Step 0: confirm builds intact
dirtyfrag prep                  # Steps 0.5 + 1: kvm group + cloud-init seed (idempotent)
dirtyfrag boot vanilla          # Step 2: launch Image-vanilla in qemu+tmux
dirtyfrag guest                 # ssh to the running guest (separate terminal)
dirtyfrag boot patched          # Step 5: same with Image-patched
dirtyfrag attach                # re-attach to qemu's tmux if you got disconnected
dirtyfrag clean                 # Step 7: kill tmux + remove seed
dirtyfrag go vanilla            # all-in-one: verify + prep + boot vanilla
dirtyfrag go patched            # all-in-one for patched
```

Typical end-to-end flow:

```bash
dirtyfrag go vanilla            # one command: verifies, preps, boots
                                # (in a second terminal:)
dirtyfrag guest                 # ssh in as tester / testpass123
                                # ... fetch PoC (Step 3), compile, run, observe ...
                                # then: sudo shutdown -h now  inside the guest
dirtyfrag go patched            # same flow, patched kernel
                                # ... rerun PoC, observe failure ...
dirtyfrag clean                 # tear-down
```

The detailed steps below remain as documentation / failure-mode
reference, but you should not need to copy-paste from them in the
happy path.

## Where to run each command

The checklist below is written for the **Mac side** — every Pi command is
prefixed with `ssh perseus@hadesllm.com '…'`.  If you're
already inside an interactive ssh session (your prompt reads
`perseus@zeus:~ $`), DROP the outer `ssh perseus@... '…'` wrapper and
run only the inner command, otherwise the Pi will try to ssh back to
itself and fail with `Permission denied (publickey)` because zeus does
not have its own pubkey in its own `authorized_keys`.

Quick test for which side you're on:

```bash
hostname    # zeus     → you are on the Pi; drop the ssh wrapper
            # Kobe24   → you are on the Mac; keep the ssh wrapper
```

## Verified state (2026-05-08 morning)

| Item | Where | Status |
|---|---|---|
| Pi mitigation `dirtyfrag.conf` | `/etc/modprobe.d/dirtyfrag.conf` on zeus | ✅ deployed; `modprobe esp4/esp6/rxrpc` correctly fails |
| Vanilla 6.12.30 source | `zeus:/home/perseus/dirtyfrag/linux-6.12.30/` | ✅ unpacked, configured |
| Patched 6.12.30 source | `zeus:/home/perseus/dirtyfrag/linux-6.12.30-patched/` | ✅ rxkad.c + esp4.c + esp6.c patched |
| Vanilla build | `zeus:.../Image-vanilla` (39 MB) | ✅ done @ 01:44 EDT (`VANILLA_BUILD_DONE_1778219073`) |
| Patched build | `zeus:.../Image-patched` (39 MB) | ✅ done @ 02:25 EDT (`PATCHED_BUILD_DONE_1778221556`) |
| Debian arm64 cloud image | `zeus:.../debian-13-arm64.qcow2` (337 MB) | ✅ downloaded |
| qemu / cloud-localds binaries | `/usr/bin/qemu-system-aarch64`, `/usr/bin/cloud-localds` | ✅ installed |
| KVM device | `/dev/kvm` (root:kvm 0660) | ✅ present |
| `perseus` in `kvm` group | `groups perseus` | ❌ NOT a member — see Step 0.5 below |
| cloud-init seed (`seed.qcow2`, `user-data`, `meta-data`) | `zeus:/home/perseus/dirtyfrag/` | ❌ not created yet — see Step 1 |
| Real diffs | `papers/dirty-frag-research/upstream-email/000{1,2,3}-*.diff` | ✅ saved |
| Email cover letter draft | `upstream-email/0000-cover-letter.md` | ✅ drafted, NOT SENT |

Free disk on `/home` (NVMe): 703 GB — plenty of room for the qemu run.

## Step 0 — re-confirm builds are intact

Pi's `tail` rejects the BSD `-3` shorthand; use POSIX `-n 3`:

```bash
ssh perseus@hadesllm.com 'tail -n 3 /home/perseus/dirtyfrag/vanilla-build.log /home/perseus/dirtyfrag/patched-build.log && ls -la /home/perseus/dirtyfrag/Image-vanilla /home/perseus/dirtyfrag/Image-patched'
#tail -n 3 /home/perseus/dirtyfrag/vanilla-build.log /home/perseus/dirtyfrag/patched-build.log && ls -la /home/perseus/dirtyfrag/Image-vanilla /home/perseus/dirtyfrag/Image-patched
```

Expected: each log ends with `..._BUILD_DONE_<unix-ts>` and both `Image-*` files are ~39 MB.  If `tail` errors with `option used in invalid context -- 3` you forgot to add `-n`.  Do NOT run a log path as a bare command (`/home/perseus/.../foo.log` on its own) — bash will try to *execute* the log file and fail with `Permission denied`; always pair the path with `tail -n N`, `cat`, or `less`.

## Step 0.5 — one-time: add `perseus` to the `kvm` group

`-accel kvm` requires R/W access to `/dev/kvm`, which is `root:kvm 0660`.  `perseus` is not in `kvm`; add it once:

```bash
ssh perseus@hadesllm.com 'sudo usermod -aG kvm perseus && id perseus'
```

Confirm `id perseus` lists `kvm` in the group set.  Then log out and back in (the existing ssh session will NOT see the new group):

```bash
exit                                                # if you are inside ssh
ssh perseus@hadesllm.com 'groups | tr " " "\n" | grep -x kvm && echo ok'
```

The second command should print `kvm` and `ok`.  Skipping this turns Step 2 into a 10-20× slower TCG run.

## Step 1 — create the cloud-init seed

Single ssh round-trip, all written into `/home/perseus/dirtyfrag/`:

```bash
ssh perseus@hadesllm.com 'bash -s' <<'EOSSH'
set -e
cd /home/perseus/dirtyfrag
cat > user-data <<'EOC'
#cloud-config
hostname: dirtyfrag-test
ssh_pwauth: True
users:
  - name: tester
    sudo: ALL=(ALL) NOPASSWD:ALL
    plain_text_passwd: testpass123
    lock_passwd: false
    shell: /bin/bash
chpasswd:
  expire: false
runcmd:
  - apt-get update
  - apt-get install -y gcc make build-essential libc6-dev
EOC
printf 'instance-id: i-dirtyfrag-test\nlocal-hostname: dirtyfrag-test\n' > meta-data
# IMPORTANT: -d qcow2 — without this flag cloud-localds writes a RAW
# ISO9660 image despite the .qcow2 extension, and qemu's
# `format=qcow2` line in Step 2 will fail with
# "Image is not in qcow2 format".  Use -d qcow2 to make the contents
# match the extension.
cloud-localds -d qcow2 seed.qcow2 user-data meta-data
qemu-img info seed.qcow2 | head -3       # should say: file format: qcow2
ls -la seed.qcow2 debian-13-arm64.qcow2
EOSSH
```

Expected output: `qemu-img info` reports `file format: qcow2`, `seed.qcow2` ~370 KB, `debian-13-arm64.qcow2` ~337 MB.

If you already created the seed without `-d qcow2`, fix it in place without regenerating user-data:

```bash
ssh perseus@hadesllm.com 'cd /home/perseus/dirtyfrag && cloud-localds -d qcow2 seed.qcow2 user-data meta-data && qemu-img info seed.qcow2 | head -3'
```

> Tip — `bash -s` over ssh is more robust than the older `ssh host <<EOSSH` form because the heredoc body is fed to `bash` on stdin, not parsed twice.

## Step 2 — boot vanilla kernel in qemu

Open a tmux/screen on the Pi (so the qemu monitor stays alive even if your ssh disconnects):

```bash
ssh -t perseus@hadesllm.com 'tmux new-session -A -s dirtyfrag'
```

Inside that tmux:

```bash
cd /home/perseus/dirtyfrag
qemu-system-aarch64 \
  -M virt -accel kvm -cpu host -m 4G -smp 2 \
  -kernel Image-vanilla \
  -append "root=/dev/vda1 rw console=ttyAMA0 net.ifnames=0" \
  -drive file=debian-13-arm64.qcow2,format=qcow2,if=none,id=hd0 \
  -device virtio-blk-device,drive=hd0 \
  -drive file=seed.qcow2,format=qcow2,if=none,id=cloudinit \
  -device virtio-blk-device,drive=cloudinit \
  -netdev user,id=net0,hostfwd=tcp::2222-:22 \
  -device virtio-net-device,netdev=net0 \
  -nographic
```

Wait for the cloud-init `runcmd` to finish (`apt-get install gcc make ...`), then on a second ssh into the Pi:

```bash
ssh -p 2222 tester@127.0.0.1     # password: testpass123
                                 # the SSH listener is on the Pi, port-forwarded into the guest
```

If the `tester` ssh hangs, cloud-init may still be running — give it 60-90 s after first prompt.

To detach the tmux without killing qemu: `Ctrl-b` then `d`.  To reattach later: `tmux attach -t dirtyfrag`.  To send Ctrl-A x to qemu (graceful shutdown) you must double the prefix because tmux owns Ctrl-A by default; use the qemu monitor instead: `Ctrl-a c` → `(qemu) quit`, *or* shut down from inside the guest with `sudo shutdown -h now`.

## Step 3 — fetch the public PoC (your hands, not mine)

I will not pull exploit code from the network.  You fetch the public Dirty Frag PoC; sources publicly disclosed since the embargo broke:

* openwall oss-security thread: <https://www.openwall.com/lists/oss-security/2026/05/07/8>
* LWN article: <https://lwn.net/Articles/1071719/>
* the author's GitHub gist (if Kim posted one) — search the openwall thread for the link

Save the PoC to `~/dirtyfrag-poc.c` (or whatever extension) **inside the qemu guest VM only** — never on the Pi host.  The guest is throw-away; the Pi is not.

## Step 4 — verify on vanilla (PoC SHOULD succeed)

Inside the guest, as user `tester` (uid=1000):

```bash
gcc -o poc dirtyfrag-poc.c
id                   # uid=1000(tester) gid=1000(tester) groups=...
./poc                # PoC runs
id                   # if uid=0(root): vanilla is exploitable, as expected
```

Note the timing and any side-output.  Then halt the VM:

```bash
sudo shutdown -h now      # from inside the guest
```

`Ctrl-A x` from the qemu console also works but only outside tmux's Ctrl-A capture.

## Step 5 — verify on patched (PoC SHOULD fail)

Same qemu command as Step 2 but `-kernel Image-patched`.  Re-ssh into the guest and re-run the PoC binary you compiled in Step 4 (it's preserved in the guest disk image):

```bash
id                   # uid=1000(tester)
./poc                # should fail/no-op
id                   # still uid=1000 — patch holds
```

Note the failure mode: hard error, hang, silent no-op?  Useful for the email writeup.

## Step 6 — finalize email + send

If both verifications pass, edit `upstream-email/0000-cover-letter.md`:

1. Replace each `[NOT VERIFIED]` line with `[VERIFIED 2026-05-08]` plus a one-line description of how it was verified.
2. Add a paragraph describing the qemu setup (arm64 / KVM on Raspberry Pi 5 / Debian 13 cloud image / kernel 6.12.30 vanilla vs patched / PoC source URL / observed behaviour on each).
3. Run the patches through `checkpatch.pl`:
   ```bash
   ssh perseus@hadesllm.com 'cd /home/perseus/dirtyfrag/linux-6.12.30-patched && ./scripts/checkpatch.pl --no-tree --strict' < /path/to/workspace/papers/dirty-frag-research/upstream-email/0001-rxrpc-rxkad-real.diff
   # repeat for 0002-xfrm-esp4-real.diff and 0003-xfrm-esp6-real.diff
   ```
   `WARNING:` lines must be addressed.  `CHECK:` lines may be ignored if they conflict with the surrounding kernel style (which is rare, but happens around `goto` labels and rxrpc's wide tables).
4. Send via `git send-email` from your **local** clone, not from the Pi — this email goes under your name from `hadesllm@proton.me`, with you reviewing each addressee:
   ```bash
   cd /path/to/workspace/papers/dirty-frag-research
   git send-email \
     --to=dhowells@redhat.com \
     --to=steffen.klassert@secunet.com \
     --cc=security@kernel.org \
     --cc=linux-distros@vs.openwall.org \
     --cc=herbert@gondor.apana.org.au \
     --cc=davem@davemloft.net \
     --cc=netdev@vger.kernel.org \
     --cc=linux-afs@lists.infradead.org \
     --cc=linux-kernel@vger.kernel.org \
     --from='Vansh Singh Ruhela <hadesllm@proton.me>' \
     upstream-email/0001-rxrpc-rxkad-real.diff \
     upstream-email/0002-xfrm-esp4-real.diff \
     upstream-email/0003-xfrm-esp6-real.diff
   ```

## Step 7 — clean up

The qemu VM is throw-away; just exit qemu (Step 4/5 already shut it down) and detach/kill the tmux:

```bash
ssh perseus@hadesllm.com 'tmux kill-session -t dirtyfrag 2>/dev/null; ls -la /home/perseus/dirtyfrag/Image-* /home/perseus/dirtyfrag/linux-6.12.30*'
```

Keep the `Image-*` files and the patched source for reference.  Optionally delete the cloud-init seed and Debian image if you need disk back:

```bash
ssh perseus@hadesllm.com 'rm -f /home/perseus/dirtyfrag/seed.qcow2 /home/perseus/dirtyfrag/user-data /home/perseus/dirtyfrag/meta-data'
# Leave debian-13-arm64.qcow2 in place if you want to re-run; it's only 337 MB.
```

(The original checklist had a malformed `shutdown -h now /home/perseus/dirtyfrag/*` here — that command makes no sense and would have errored.  Ignore.)

## Failure-mode decision tree

| Symptom | What it means | Action |
|---|---|---|
| `tail: option used in invalid context -- 3` | BSD shorthand on Pi `tail` | Re-run with `tail -n 3` |
| `Permission denied` after typing a log path | You ran the file as a command | Prefix with `tail -n N`, `cat`, or `less` |
| qemu: `Could not access KVM kernel module: Permission denied` | `perseus` not in `kvm` group, OR you didn't re-login | Re-do Step 0.5 + open a fresh ssh |
| qemu: `Image is not in qcow2 format` (on `seed.qcow2`) | `cloud-localds` ran without `-d qcow2`, so the file is raw ISO9660 despite the extension | Re-run `cloud-localds -d qcow2 seed.qcow2 user-data meta-data`; verify with `qemu-img info seed.qcow2` |
| Kernel panic: `VFS: Unable to mount root fs on unknown-block(254,1)` | `virtio-blk-device` (MMIO transport) didn't get picked up by the device tree under `-M virt -accel kvm` — `vda` never appears | Use the PCI variant: `-device virtio-blk-pci` and `-device virtio-net-pci` (no other config changes needed; `CONFIG_PCI_HOST_GENERIC=y` is already set).  The `dirtyfrag` driver was updated 2026-05-08 to use PCI by default |
| qemu exits immediately with `KVM does not support GICv3 emulation` | Pi 5 is GICv2; forcing `gic-version=3` makes KVM refuse | Use `gic-version=host` (or just drop the option).  `dirtyfrag` already uses `gic-version=host` |
| Login prompt rejects `tester` / `testpass123`, hostname is `localhost`, cloud-init log says `Datasource DataSourceNone. Used fallback datasource` | Debian 13 cloud image's `datasource_list` doesn't include NoCloud, so cloud-init never read our seed disk and never created the `tester` user | Append `ds=nocloud` to the kernel cmdline — `dirtyfrag` does this automatically as of 2026-05-08 |
| Patched kernel won't boot | The patch broke kernel init | Don't send.  Revisit the patch sketches; the bounce-buffer pattern in PATCH 1 is the most likely suspect (`skb_store_bits` semantics may be subtler than I sketched) |
| Vanilla kernel fails to exploit | qemu config too minimal — modules not actually loaded? | Inside guest: `lsmod \| grep -E 'esp4\|esp6\|rxrpc'`.  If absent, `modprobe esp4 esp6 rxrpc`.  If still not exploitable: maybe qemu virt doesn't expose the affected page-splice surface; try without `-accel kvm` |
| PoC requires unprivileged userns | Some kernel exploits need `kernel.unprivileged_userns_clone=1` | Set inside guest: `sudo sysctl -w kernel.unprivileged_userns_clone=1`, retest |
| Both kernels behave the same (PoC fails on vanilla too) | Likely qemu config issue, not patch effectiveness | Try without `-accel kvm` (slower TCG path is more faithful to UP/SMP race timing) |
| Upstream patch lands during our work | They beat us | Read theirs, compare with ours, decide whether to send anyway as cross-check (low value) or stand down (also fine) |

## What to do if upstream has already shipped

Check `https://security-tracker.debian.org/tracker/` and `https://lwn.net/Articles/1071719/` (or follow-up LWN articles) for "Dirty Frag" status.  If Debian has shipped a DSA patching this:

1. Don't send our patches — upstream's are more authoritative.
2. On the Pi: `sudo apt-get update && sudo apt-get full-upgrade && sudo reboot`
3. After reboot: `sudo rm /etc/modprobe.d/dirtyfrag.conf` to restore IPsec/RxRPC.
4. Mark task #43 as completed.
5. Save our research as a "what we drafted independently" reference under `papers/dirty-frag-research/`.

## Phase tracker (4-phase plan)

- [x] **Phase 1** — Pi mitigation deployed
- [x] **Phase 2** — Vanilla + patched 6.12.30 kernels built (✅ this morning)
- [ ] **Phase 3** — qemu verify (you are here): Steps 0.5 → 5
- [ ] **Phase 4** — checkpatch + email upstream: Step 6
