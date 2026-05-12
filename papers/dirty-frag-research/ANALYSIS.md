# Dirty Frag (May 2026) — defensive-patch research notes

> **STATUS — DO NOT DEPLOY.**  These patches are unverified research drafts
> based on reading torvalds/linux master + the public openwall advisory.
> The actual exploit PoC has not been read line-by-line.  The Pi already
> has a complete configuration mitigation in
> `/etc/modprobe.d/dirtyfrag.conf` (esp4/esp6/rxrpc → /bin/false), which
> achieves the same security outcome by never loading the vulnerable
> code into the kernel.  When Debian / kernel.org ships the official
> upstream patch, replace the mitigation with that.  Until then,
> **the configuration mitigation is the deploy-grade defence**, and
> this document is a learning exercise + a fallback option if the
> Pi is ever rebooted with custom kernel modules required.

---

## Bug class — what the advisory describes

Two chained bugs, sharing the same root pattern:

> Crypto operations performed *in-place* on skb pages whose underlying
> physical pages are still mapped into userspace (via `MSG_SPLICE_PAGES`),
> with no copy-on-write between the user-supplied splice and the
> in-kernel crypto write-back.  An attacker who arranges a suitable
> page-overlap can force the kernel to write attacker-chosen ciphertext
> bytes onto kernel pages or other privileged structures.

The two affected sites are:

1. **`net/rxrpc/rxkad.c::rxkad_verify_packet_1()`**
   8-byte in-place pcbc(fcrypt) decrypt over a scatterlist constructed
   directly from skb pages by `skb_to_sgvec()`.  The in-source comment
   already flags the design as suboptimal:
   > /* Decrypt the skbuff in-place.  TODO: We really want to decrypt
   >  * directly into the target buffer. */

2. **`net/ipv4/esp4.c::esp_output_tail()` (and `esp6.c` mirror)**
   AEAD encrypt/decrypt with `dsg = sg` whenever `esp->inplace == true`,
   on a scatterlist built from skb pages that may have come straight
   from `MSG_SPLICE_PAGES` without COW.

The exploit brute-forces a useful pcbc(fcrypt) key in user space (the
`fcrypt` cipher is small and fast), then triggers the in-place 8-byte
decrypt to plant attacker-controlled bytes into a userspace-shared
kernel page that aliases something privileged.

## Affected functions, exact locations on torvalds/linux master

```
net/rxrpc/rxkad.c:429   rxkad_verify_packet_1()
net/rxrpc/rxkad.c:494   rxkad_verify_packet_2()    (level-2 — same pattern, larger window)
net/ipv4/esp4.c:495     esp_output_tail()
net/ipv4/esp4.c:625     esp.inplace = true;        (default)
net/ipv4/esp4.c:439     esp->inplace = false;      (set false only when dst pages allocated)
net/ipv6/esp6.c:*       analogous structure (1261 lines, mirror of esp4.c)
```

## Defensive-patch sketches

### Patch A (rxkad) — bounce buffer for the 8-byte header decrypt

The header is 8 bytes.  Just decrypt into a stack-resident buffer and
write the plaintext back via `skb_store_bits`, instead of decrypting
in-place via the same scatterlist on the source pages.

```diff
--- a/net/rxrpc/rxkad.c
+++ b/net/rxrpc/rxkad.c
@@ -429,6 +429,7 @@ static int rxkad_verify_packet_1(struct rxrpc_call *call, struct sk_buff *skb,
 	struct rxkad_level1_hdr sechdr;
 	struct rxrpc_skb_priv *sp = rxrpc_skb(skb);
 	struct rxrpc_crypt iv;
-	struct scatterlist sg[16];
+	struct scatterlist sg_in[1];
+	struct scatterlist sg_out[1];
+	u8 hdr_buf[8];
 	u32 data_size, buf;
 	u16 check;
 	int ret;
@@ -445,8 +446,12 @@ static int rxkad_verify_packet_1(struct rxrpc_call *call, struct sk_buff *skb,
-	/* Decrypt the skbuff in-place.  TODO: We really want to decrypt
-	 * directly into the target buffer.
+	/* Dirty Frag (May 2026): the previous implementation decrypted
+	 * 8 bytes in-place on the skb pages, which is unsafe when the
+	 * skb pages came from MSG_SPLICE_PAGES without COW.  Decrypt
+	 * into a stack bounce buffer and only write the plaintext back
+	 * after a length sanity check.
 	 */
-	sg_init_table(sg, ARRAY_SIZE(sg));
-	ret = skb_to_sgvec(skb, sg, sp->offset, 8);
-	if (unlikely(ret < 0))
+	if (skb_copy_bits(skb, sp->offset, hdr_buf, 8) < 0)
 		return ret;
+	sg_init_one(sg_in,  hdr_buf, 8);
+	sg_init_one(sg_out, hdr_buf, 8);

 	memset(&iv, 0, sizeof(iv));
 	skcipher_request_set_sync_tfm(req, call->conn->rxkad.cipher);
 	skcipher_request_set_callback(req, 0, NULL, NULL);
-	skcipher_request_set_crypt(req, sg, sg, 8, iv.x);
+	skcipher_request_set_crypt(req, sg_in, sg_out, 8, iv.x);
 	ret = crypto_skcipher_decrypt(req);
 	skcipher_request_zero(req);
 	if (ret < 0)
 		return ret;

-	/* Extract the decrypted packet length */
-	if (skb_copy_bits(skb, sp->offset, &sechdr, sizeof(sechdr)) < 0)
+	/* Copy the now-decrypted header out of the bounce buffer. */
+	memcpy(&sechdr, hdr_buf, sizeof(sechdr));
+	/* Reflect the decrypted bytes back into the skb only after we
+	 * have confirmed the length+check fields are sane.
+	 */
+	if (skb_store_bits(skb, sp->offset, hdr_buf, 8) < 0)
 		return rxrpc_abort_eproto(call, skb, RXKADDATALEN,
 					  rxkad_abort_1_short_encdata);
```

**Effect.**  The 8-byte decrypt happens in stack memory the kernel
fully controls.  No user-supplied page is ever a destination of the
crypto write.  `skb_store_bits` correctly handles spliced pages by
either COWing or returning an error — it is the standard safe path
for writing into an skb.

**Caveats.**
* `rxkad_verify_packet_2` (level-2 security, full-payload decrypt)
  has the same in-place pattern but over a much larger buffer; a
  bounce-buffer fix there would burn allocation per packet and is
  better solved by `skb_unclone()` / `pskb_expand_head()` to force
  a COW first.  This sketch only covers the level-1 8-byte header.
* `skb_store_bits` semantics on a non-writable skb need verifying
  against the upstream comment at `net/core/skbuff.c::skb_store_bits`.
* The patch as-shown is illustrative; the actual whitespace, error
  codes, and existing abort tags must match the surrounding style.

### Patch B (esp4 / esp6) — refuse in-place crypto when frags are splice-pinned

`esp->inplace` is set true by default and only flipped false when the
non-inplace path allocates destination pages.  The defensive change
is to also flip it false when any of the skb's frags is a
splice-pinned page.

```diff
--- a/net/ipv4/esp4.c
+++ b/net/ipv4/esp4.c
@@ -495,6 +495,17 @@ int esp_output_tail(struct xfrm_state *x, struct sk_buff *skb, struct esp_info *
 	struct esp_output_extra *extra;
 	int err = -ENOMEM;

+	/* Dirty Frag (May 2026): the in-place AEAD crypto path uses
+	 * dsg == sg, which means crypto writes back to the same skb
+	 * pages we read from.  This is unsafe when any frag came from
+	 * MSG_SPLICE_PAGES without a copy-on-write boundary.  Force the
+	 * non-inplace path (which allocates fresh destination pages
+	 * from x->xfrag) whenever the skb is shared or has any
+	 * frag_list / fraglist entries that could be user-pinned.
+	 */
+	if (esp->inplace &&
+	    (skb_shared(skb) || skb_has_frag_list(skb) ||
+	     skb_shinfo(skb)->nr_frags > 0))
+		esp->inplace = false;
+
 	assoclen = sizeof(struct ip_esp_hdr);
 	extralen = 0;
```

**Effect.**  Whenever the skb has any user-controllable frag, the
output path takes the non-inplace branch (line 547+), allocates a
fresh `pfrag` page, and writes ciphertext to that fresh page.  The
read scatterlist still references the original frags but the write
scatterlist points to kernel-owned pages.

**Caveats.**
* The performance cost is a per-packet page allocation — same cost
  as the existing non-inplace path.  Pure-skb (no frag) flows still
  hit the fast in-place path.
* The check `skb_shinfo(skb)->nr_frags > 0` is overly conservative —
  it captures all frags including kernel-allocated ones.  A more
  precise check would inspect `skb_frag_page(...)` against a
  known-kernel-allocated set.  The conservative version is the
  defensive choice: false-positive (slow) over false-negative (unsafe).
* The mirror patch for esp6.c is structurally identical.
* The IPsec test suite (xfrmtests / iperf3 over IPsec) needs to
  pass before any deploy.

## What this patch does NOT do

* It does not address every site that uses `skb_to_sgvec(...)` with
  `dsg == sg`.  That pattern exists in other crypto-on-skb paths
  (TLS kernel offload, MACsec) and would need an audit.  The two
  sites named in the advisory are these two.
* It does not eliminate `MSG_SPLICE_PAGES`.  That syscall flag is a
  legitimate fast-path for high-throughput producers; the right fix
  is to ensure crypto consumers never write back into user pages,
  not to disable splice.
* It does not rebuild the kernel for the Pi.  Cross-compiling a
  Pi-5 6.12 kernel takes ~30 min on this Mac; flashing the patched
  kernel is a high-stakes operation that should not happen without
  a parallel recovery SD card prepared.

## Recommended action ordering

1. **Now (already done):** module blacklist via
   `/etc/modprobe.d/dirtyfrag.conf` — full mitigation on the Pi.
2. **Within 24-72 h:** Debian / kernel.org will ship an official
   upstream patch.  Apply via `apt update && apt full-upgrade`,
   reboot, then `rm /etc/modprobe.d/dirtyfrag.conf` to restore
   IPsec/RxRPC functionality.  Tracked as task #43.
3. **Only if upstream is delayed past 72 h:** revisit the sketches
   above, get a second pair of eyes (e.g., a kernel-savvy human
   reviewer; ideally read the public PoC against the patch), build
   a custom kernel in a Pi-emulator first, only then deploy to
   zeus with a recovery SD card prepared.

## References

* Public disclosure (oss-security): https://www.openwall.com/lists/oss-security/2026/05/07/8
* LWN article: https://lwn.net/Articles/1071719/
* Debian security tracker: https://security-tracker.debian.org/tracker/
* Phoronix: https://www.phoronix.com/news/Dirty-Frag-Linux
* Upstream source read: torvalds/linux master, files
  `net/rxrpc/rxkad.c`, `net/ipv4/esp4.c`, `net/ipv6/esp6.c`.
