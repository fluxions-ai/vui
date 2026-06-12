# Talk from your phone

Mobile browsers refuse mic access over plain HTTP, so opening `http://<lan-ip>:8080` from your phone won't work — you need HTTPS. The other complication is **WebRTC**: the browser UI uses host-only ICE (`iceServers: []` in `index.html`), so media flows peer-to-peer to the server's LAN IP — *not* through any HTTPS tunnel. That makes the right choice depend on where your phone is:

| Where's the phone? | Easiest path | Why |
|---|---|---|
| **Same Wi-Fi as the server** | cloudflared (below) — covers HTTPS, WebRTC goes direct on the LAN | One command, no account |
| **Cellular / away from home** | [Tailscale](https://tailscale.com) — phone + server on the same tailnet | Proper L3 routing, host-candidate WebRTC just works, no TURN server needed |
| **Custom client, anywhere** | Build against `/v1/realtime` ([`realtime-api.md`](realtime-api.md)) | All-WebSocket, no WebRTC — traverses any HTTPS proxy |

## cloudflared (same Wi-Fi)

[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) gives you automatic HTTPS with no account or port-forwarding:

```sh
brew install cloudflared              # macOS
# Linux: grab the single binary from https://github.com/cloudflare/cloudflared/releases

cloudflared tunnel --url http://localhost:8080
```

cloudflared prints a `https://<random>.trycloudflare.com` URL — open it on your phone (on the same Wi-Fi as the server), allow mic access, and you're in. The page, `/ws`, `/v1/realtime`, and the `/offer` SDP exchange all flow through the tunnel; WebRTC media takes a peer-to-peer shortcut on the LAN.

**Stable URL?** `cloudflared tunnel login` + `cloudflared tunnel create vui` lets you point a hostname (e.g. `vui.yourdomain.com`) at `http://localhost:8080` — see Cloudflare's [named tunnel docs](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-remote-tunnel/).

## Tailscale (anywhere)

When you're off the LAN, cloudflared isn't enough — its tunnel is TCP-only, so it can't carry WebRTC's UDP media. Easiest fix is to put your phone and server on the same tailnet:

1. Install [Tailscale](https://tailscale.com) on the server and the phone (free for personal use).
2. On the server, expose HTTPS to the tailnet: `tailscale serve --bg --https=443 http://localhost:8080`.
3. On the phone (signed into the same tailnet), open `https://<server-hostname>.<tailnet>.ts.net`.

You get a valid TLS cert from Tailscale, the server's LAN IP becomes routable from the phone over the tailnet, and host-candidate WebRTC connects without any TURN/STUN config. No public URL, no auth concern, works on cellular.

## Gotchas (both paths)

- **No auth on Vui** — anyone with reachability to `:8080` can talk to your assistant. The cloudflared URL is unguessable and dies on restart, and Tailscale is private to your tailnet, but treat both as you would `ssh` access.
- **Single-tenant.** `/offer` and `/v1/realtime` accept one client at a time (HTTP 409 on the second). Close the desktop tab before joining from the phone.
- **Off-LAN with cloudflared?** Page loads but WebRTC fails to connect. Either switch to Tailscale (above), drive `/v1/realtime` from a custom WebSocket client, or add a public TURN server to `iceServers` in `index.html`.
