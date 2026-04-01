# SwiftNVH — Right-Click Context Menu Reference

Right-clicking on the graph canvas opens a context-sensitive menu depending on **where** you click:

| Region | Menu Shown |
|--------|-----------|
| Plot area (curves) | Graph Area Menu |
| Y-axis label/ticks | Y-Axis Menu |
| X-axis label/ticks | X-Axis Menu |

---

## Apply to All Graphs (header checkbox)

Every menu begins with a checkbox **"Apply to All Graphs"**.

- **Unchecked (default)** — action applies only to the currently active subplot.
- **Checked** — action is applied to every subplot in the current branch simultaneously.

---

## Graph Area Menu

Right-click anywhere inside the plot area (on the curves or background).

### Cursors

| Action | Description |
|--------|-------------|
| **Add Single Cursor** | Places a single vertical cursor line; shows Y value at cursor X position. |
| **Add Double Cursor** | Places two cursor lines; shows delta-X and delta-Y between them. |
| **Remove all Cursors** | Removes all cursor lines from the current plot. |
| **Snap to Data** *(checkable)* | When enabled, cursors snap to the nearest data point rather than interpolating. |

---

### Curves

Sub-menu for visibility and layout of individual curves.

| Action | Description |
|--------|-------------|
| **Show All** | Makes every curve visible. |
| **Hide All** | Hides every curve (useful for isolating one manually). |
| *[Direction toggles: X / Y / Z / Rx / Ry / Rz / SPL]* | In **overlay mode**, checkboxes to show/hide each vibration direction globally. |
| *[Per-curve checkboxes]* | Toggle each loaded curve on/off individually. |

---

### Show Only

Sub-menu listing every curve. Clicking a name hides all others and shows only that curve.

---

### Select all Curves / Remove all Curves

Shortcuts to show or hide every curve (same as Curves → Show All / Hide All).

---

### Change Color

Sub-menu listing each curve by name. Clicking opens a colour-picker dialog to change that curve's line colour.

---

### Line Width

| Sub-menu / Action | Description |
|-------------------|-------------|
| **All Curves → 0.5 / 1.0 / 1.5 / 2.0 / 3.0** | Sets the same line width for every curve at once. |
| **[Curve name] → 0.5 / 1.0 / 1.5 / 2.0 / 3.0** | Sets line width for a single curve. |

---

### Line Style

| Sub-menu / Action | Description |
|-------------------|-------------|
| **All Curves → Solid / Dashed / Dotted** | Sets the same line style for every curve. |
| **[Curve name] → Solid / Dashed / Dotted** | Sets line style for a single curve. |

---

### Legend

| Action | Description |
|--------|-------------|
| **Legend** *(checkable)* | Show or hide the legend box. |
| **Legend Position → Best / Upper Right / Upper Left / Lower Left / Lower Right / Right / Center Left / Center Right** | Move the legend to a standard matplotlib anchor position. |

---

### Labels

| Action | Description |
|--------|-------------|
| **Edit Title...** | Opens an input dialog to change the subplot title. |
| **Edit X Label...** | Opens an input dialog to change the X-axis label text. |
| **Edit Y Label...** | Opens an input dialog to change the Y-axis label text. |
| **Rename Curve... → "[Curve name]"...** | Opens an input dialog to rename a specific curve in the legend (affects all sibling branches). |

---

### View

| Action | Description |
|--------|-------------|
| **Autoscale** | Resets both axes to fit all visible data (removes any fixed limits). |
| **Toggle Grid** | Shows or hides the background grid lines. |

---

### Page Grouping *(radio buttons)*

Controls how multiple subplots are arranged across pages.

| Option | Description |
|--------|-------------|
| **Group by Direction (X / Y / Z)** | Each page shows one vibration direction; mounts are rows. *(default)* |
| **Group by Node (Mount)** | Each page shows one mount; directions are rows. |

---

### Target Line

| Action | Description |
|--------|-------------|
| **Add Mobility Target...** | Opens a dialog to add a horizontal reference line (e.g. velocity limit) with a custom value and unit. |
| **Remove All Targets** *(disabled when no targets)* | Deletes all target/limit lines from the current plot. LMS targets loaded from the database are also removed. |

---

### Copy / Save

| Action | Description |
|--------|-------------|
| **Copy as → Bitmap** | Copies the current plot as a PNG image to the clipboard (paste into Word/PowerPoint/etc.). |
| **Save PNG...** | Opens a file dialog to save the current view as a PNG file. |
| **Export Excel...** | Exports ALL branches (not just the current view) to an Excel file; each branch × flow is a separate sheet. |
| **Copy Values** | Copies the raw X/Y data of all visible curves to the clipboard as tab-separated text (paste into Excel). |

---

## Y-Axis Menu

Right-click on the **Y-axis area** (tick labels or axis title).

### Limits

| Action | Description |
|--------|-------------|
| **Free** | Removes any fixed limits; Y-axis auto-scales to the data. |
| **Optimized** | Computes nice round limits that encompass all visible data. |
| **Fixed...** | Opens a dialog to enter exact Y-min and Y-max values. |

---

### Scale *(radio buttons)*

| Option | Description |
|--------|-------------|
| **Linear** | Standard linear Y-axis. |
| **Log** | Logarithmic Y-axis (useful for wide-range data). |
| **dB (ISO)** | Converts data to ISO dB: `20·log₁₀(val / ref)`, ref = 1 µm/s², peak. |
| **dB RMS** | Converts data to Team RMS dB: `20·log₁₀(val / √2)`, ref = 1 m/s², RMS. Offset is +123 dB vs ISO. |

---

### Unit

Changes how the Y data is displayed. Options depend on what physical quantity was detected (e.g. Acceleration, Pressure, Velocity).

| Sub-menu / Action | Description |
|-------------------|-------------|
| *[Same-group units as radio buttons]* | Switch between units in the same physical group (e.g. `m/s²` ↔ `mm/s²` ↔ `g`). |
| *[dB modes: dB / dB(A) / dB(B) / dB(C)]* | Show data in decibels with A/B/C frequency weighting. |
| *[RMS dB modes]* | Show data in RMS-referenced dB with optional weighting. |
| **Change Data Type... → [Group → Unit]** | Override the detected input unit (e.g. force the data to be interpreted as Pressure instead of Acceleration). |
| **No conversion** | Display raw data values with no unit conversion. |

---

### Processing

| Sub-menu / Action | Description |
|-------------------|-------------|
| **Peak** *(radio)* | Data is shown as peak amplitude (raw simulation values). |
| **RMS (÷√2)** *(radio)* | Data is divided by √2 before display (converts peak → RMS). |
| **Weighting → None (Linear) / dB (flat) / A-weighted / B-weighted / C-weighted** | Apply frequency weighting to the dB display. |

---

### Visible *(checkable)*

Show or hide the Y-axis (tick labels, gridlines, and title). Useful for clean presentation exports.

---

## X-Axis Menu

Right-click on the **X-axis area** (tick labels or "Frequency [Hz]" label).

### Limits

| Action | Description |
|--------|-------------|
| **Free** | Removes fixed limits; X-axis auto-scales to the data range. |
| **Optimized** | Computes nice round limits encompassing all data. |
| **Fixed...** | Opens a dialog to enter exact X-min and X-max values. |

### Visible *(checkable)*

Show or hide the X-axis (tick labels and title).

---

## Keyboard / Mouse Shortcuts (within the plot area)

| Shortcut | Action |
|----------|--------|
| **Shift + Left-click** on a subplot | Opens the subplot in a **Popout Window** (full-screen detail view). Closing the popout syncs axis limits back to the main GUI. |
| **Double-click** on Y- or X-axis | Opens the **Fixed Limits** dialog directly (same as Limits → Fixed...). |

---

## Notes

- **Apply to All Graphs** works for most formatting actions (scale, limits, style, colour, grid) but **not** for label edits (each subplot can have its own title/labels).
- Target lines added via **Add Mobility Target** are stored in the branch's `plot_settings` and survive branch switching. LMS targets from the database are re-attached automatically on each plot cycle.
- **Export Excel** always exports every branch, regardless of which is currently displayed.
