# DRAFT cover letter — Dirty Frag mitigation patches (rxrpc + xfrm)

> **STATUS — DO NOT SEND.**  This is a staging draft.  Send only
> after the patches in 0001-rxrpc-...patch and 0002-xfrm-...patch
> are confirmed to (a) compile cleanly against vanilla 6.12 and
> (b) defeat the public PoC in an isolated qemu VM.  After
> verification, replace the "[NOT VERIFIED]" markers with the
> verification methodology and rerun the diffs through `checkpatch.pl`.

---

To:        David Howells <dhowells@redhat.com>,
           Steffen Klassert <steffen.klassert@secunet.com>
Cc:        security@kernel.org,
           linux-distros@vs.openwall.org,
           Herbert Xu <herbert@gondor.apana.org.au>,
           David S. Miller <davem@davemloft.net>,
           netdev@vger.kernel.org,
           linux-afs@lists.infradead.org,
           linux-kernel@vger.kernel.org
Subject:   [PATCH 0/2] Dirty Frag (oss-security 2026-05-07): in-place
           crypto on splice-pinned skb pages — proposed defensive patches
From:      Vansh Singh Ruhela <hadesllm@proton.me>
Date:      <set by git send-email at send time>

---

Hi David, Steffen,

We are aware you are very likely already working on this; this
mail is a *cross-check*, not a replacement.  We read the public
oss-security advisory and the upstream `net/rxrpc/rxkad.c` and
`net/ipv4/esp4.c` source, identified the same bug class flagged in
the advisory, and drafted two minimal defensive patches.  Please
take this for what it is worth — corroborating analysis from
external eyes — and discard if it conflicts with the work you have
underway.

Bug class (per oss-security 2026-05-07/8):
crypto operations performed *in-place* on skb pages whose
underlying physical pages are still mapped into userspace via
`MSG_SPLICE_PAGES`, with no copy-on-write between the user-supplied
splice and the kernel crypto write-back.  An attacker who arranges
suitable page-overlap forces the kernel to write attacker-chosen
ciphertext bytes onto pages they share with privileged structures.

Two affected sites, two patches:

  PATCH 1 — net/rxrpc/rxkad.c::rxkad_verify_packet_1()
            8-byte in-place pcbc(fcrypt) decrypt over a scatterlist
            built from skb pages.  The in-source comment already
            flags the design as suboptimal.  Fix: decrypt into a
            stack-resident bounce buffer and write the plaintext
            back via `skb_store_bits` only after the length /
            checksum sanity check.

  PATCH 2 — net/ipv4/esp4.c::esp_output_tail() (+ esp6.c mirror)
            AEAD encrypt/decrypt with `dsg = sg` whenever
            `esp->inplace == true`, on a scatterlist that may
            include splice-pinned frags.  Fix: force `inplace =
            false` whenever the skb has any frag (`nr_frags > 0`),
            shared status, or frag_list.

Verification status (current):

  [VERIFIED 2026-05-08 01:25] passes scripts/checkpatch.pl cleanly:
                              0 errors, 1 warning per diff (the warning
                              is "Missing commit description", artifact
                              of the unified-diff-vs-git-format-patch
                              format and resolved when these become
                              git-format-patch posts).
  [VERIFIED 2026-05-08 01:30] standalone compile of the 3 patched .c
                              files (rxkad.c, esp4.c, esp6.c) against
                              vanilla 6.12.30 + arm64 defconfig +
                              CONFIG_RXKAD=y, INET_ESP=y, INET6_ESP=y,
                              XFRM_USER=y, IPV6=y: zero warnings, zero
                              errors.  Object files produced
                              (rxkad.o 87.6 KB, esp4.o 125 KB,
                              esp6.o 129 KB).  An initial build flagged
                              an unused-variable warning on the now-dead
                              `int ret;` declaration in rxkad's
                              variable-block; that has been fixed in
                              0001-rxrpc-rxkad-real.diff.
  [PENDING]                   passes existing kernel selftest suite for
                              IPsec / rxrpc
  [PENDING]                   defeats the public PoC released 2026-05-08
                              (qemu-aarch64 vs vanilla kernel, then vs
                              patched kernel) — runs after both kernels
                              built
  [NOT TESTED]                no performance regression on IPsec fast
                              path (would require IPsec throughput
                              testing not currently set up)

We will follow up with verification results once we have run the
public PoC against an isolated qemu-aarch64 VM with both vanilla
6.12.30 and patched kernels.  In the meantime, the analysis and
diff sketches may be useful if your in-flight patch differs in
shape from ours.

Caveats we already know about:

  * PATCH 1 only covers `rxkad_verify_packet_1` (level-1 security,
    8-byte header).  `rxkad_verify_packet_2` (level-2, full payload)
    has the same in-place pattern over a much larger buffer and
    cannot use a stack bounce; it needs `skb_unclone()` /
    `pskb_expand_head()` first.  We have not drafted a patch for
    level-2 yet.

  * PATCH 2 is conservative — it forces the non-inplace allocation
    path whenever any frag exists, which is over-broad.  A more
    precise check would be `skb_frag_is_pfmemalloc()` against a
    known-kernel-allocated set, which we did not implement.  The
    conservative version trades fast-path performance for safety.

  * Both patches are synced against torvalds/linux master HEAD
    f1cf6263... (snapshot date 2026-05-08).  Backports to stable
    branches are straightforward but not included here.

Thanks for your time — and apologies in advance if this duplicates
work you have already shipped.  We are mitigating locally with the
modprobe blacklist that's been circulating
(`install esp4 /bin/false`, etc.) until the official patch lands.

Best,
Vansh Singh Ruhela
HADES-LLM

---

Co-developed-by: Yoda <noreply@anthropic.com>

Yoda is the project's persistent-memory analytic agent (Claude Opus
4.7).  All patch hypotheses originated from human-AI dialogue; the
Signed-off-by below reflects human responsibility for review,
testing, and submission.
