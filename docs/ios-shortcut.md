# iOS Shortcut: Card Scanner

One-tap card scanning from your iPhone. Take a photo (or pick from gallery),
upload it to the card scanner server, and see the identified card name + price.

**Prerequisites**: The card scanner server must be running and reachable from
your phone (see "Network Access" at the bottom).

---

## Step-by-Step Setup

Open the **Shortcuts** app on your iPhone and tap the **+** button in the
top-right corner to create a new shortcut.

### Action 1: Choose Photo Source

Add a **"Choose from Menu"** action with two options:

| Setting         | Value                  |
|-----------------|------------------------|
| Prompt          | `Card Scanner`         |
| Option 1        | `Take Photo`           |
| Option 2        | `Pick from Gallery`    |

### Action 2a: Take Photo (inside the "Take Photo" branch)

Add a **"Take Photo"** action.

| Setting              | Value  |
|----------------------|--------|
| Show Camera Preview  | On     |

### Action 2b: Pick from Gallery (inside the "Pick from Gallery" branch)

Add a **"Select Photos"** action.

| Setting              | Value  |
|----------------------|--------|
| Select Multiple      | Off    |

### Action 3: Set Variable (after "End Menu")

Add a **"Set Variable"** action.

| Setting   | Value                                      |
|-----------|--------------------------------------------|
| Variable  | `CardPhoto`                                |
| Input     | (tap and select **Menu Result** from above)|

This stores whichever image was produced (camera or gallery) into a single
variable we can reference later.

### Action 4: Upload the Image

Add a **"Get Contents of URL"** action. This is the core of the shortcut.
Configure it exactly as follows:

| Setting        | Value                               |
|----------------|-------------------------------------|
| URL            | `http://<YOUR-SERVER-IP>:8888/scan` |
| Method         | **POST**                            |
| Request Body   | **Form**                            |

After selecting **Form** as the request body type, tap **"Add New Field"**:

| Field Setting | Value                |
|---------------|----------------------|
| Type          | **File**             |
| Key           | `image`              |
| Value         | select `CardPhoto` variable (tap the field and pick the variable) |

Leave Headers empty -- Shortcuts handles `Content-Type: multipart/form-data`
automatically.

### Action 5: Parse the JSON Response

Add a **"Get Dictionary from Input"** action.

| Setting | Value                            |
|---------|----------------------------------|
| Input   | Contents of URL (auto-filled)    |

### Action 6: Extract Card Name

Add a **"Get Dictionary Value"** action.

| Setting    | Value          |
|------------|----------------|
| Dictionary | Dictionary     |
| Key        | `card_name`    |

Then add a **"Set Variable"** action:

| Setting   | Value       |
|-----------|-------------|
| Variable  | `CardName`  |
| Input     | Dictionary Value |

### Action 7: Extract Price

Add another **"Get Dictionary Value"** action.

| Setting    | Value          |
|------------|----------------|
| Dictionary | Dictionary (from step 5) |
| Key        | `price`        |

Then add a **"Set Variable"** action:

| Setting   | Value      |
|-----------|------------|
| Variable  | `Price`    |
| Input     | Dictionary Value |

### Action 8: Show the Result

Add a **"Show Alert"** action (or "Show Result" if you prefer).

| Setting | Value                         |
|---------|-------------------------------|
| Title   | `Card Identified`             |
| Message | `CardName` `$` `Price`        |

To build the message: type the text, then tap where you want the variable
inserted and select `CardName` and `Price` from the variable picker. It should
read something like:

```
[CardName]
Price: $[Price]
```

### Action 9 (Optional): Show Full JSON

If you want a "debug mode" that shows the raw server response, add a
**"Quick Look"** action right after the "Get Contents of URL" step, with
its input set to **Contents of URL**. You can toggle this on/off by
disabling the action (long press > Disable).

---

## Rename and Add to Home Screen

1. Tap the **dropdown arrow** at the top of the shortcut (next to the name).
2. Tap **Rename** and call it something like **Scan Card**.
3. Tap the dropdown again and select **Choose Icon** -- pick a camera or
   card-related icon and color.
4. Tap the dropdown again and select **"Add to Home Screen"**.
5. Confirm. A one-tap icon now appears on your home screen.

Tapping it opens the camera/gallery picker, uploads the photo, and shows the
card name and price in a few seconds.

---

## Expected Server Response Format

The shortcut assumes the `/scan` endpoint returns JSON like:

```json
{
  "card_name": "Charizard ex",
  "set_name": "Obsidian Flames",
  "card_id": "sv3-125/holofoil",
  "confidence": 0.94,
  "price": 42.50
}
```

Adjust the dictionary keys in steps 6-7 if your server uses different field
names.

---

## Network Access (Important)

Your phone must be able to reach the server at `http://<ip>:8888`. This works
automatically when both devices are on the same Wi-Fi network. For access
**outside your home network**, you have three options:

1. **Tailscale** (easiest) -- Install Tailscale on both the server and your
   iPhone. Use the Tailscale IP (e.g. `100.x.y.z`) as the server address in
   the shortcut. Free for personal use, zero configuration.

2. **WireGuard VPN** -- Set up a WireGuard server on your network and import
   the config into the WireGuard iOS app. Use your server's LAN IP in the
   shortcut. More setup but no third-party accounts.

3. **Cloudflare Tunnel** -- Run `cloudflared tunnel` on the server to expose
   it at a public URL (e.g. `https://cards.yourdomain.com/scan`). No VPN
   needed on the phone, but requires a domain pointed at Cloudflare. Use the
   public HTTPS URL in the shortcut instead of the LAN IP.

Any of these three work well. Tailscale is the lowest-friction option for most
people.
