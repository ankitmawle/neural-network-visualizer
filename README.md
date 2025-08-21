# Neural Network Visualizer for RISC-V Edge AI Workshop using VSDSquadron Pro

Interactive neural network visualizer for edge AI workflows. Draw directly on a pixel grid, load quantized models exported as C header files, and watch activations flow through the network in real time. Built for the VSDSquadron Pro RISC-V Edge AI Workshop.

Developed by [Ankit Mawle](https://www.linkedin.com/in/ankitmawle/)

Repository: `https://github.com/ankitmawle/neural-network-visualizer`



## Features

- Quantized C header parsing (e.g., 4-bit symmetric) with packed `uint32_t` weights
- Dynamic architecture support:
  - 2 hidden layers: Input → Hidden1 → Hidden2 → Output
  - 3 hidden layers: Input → Hidden1 → Hidden2 → Hidden3 → Output
- Automatic class labels if missing: "0".."N-1" based on output size
- Forward pass with layer-norm and ReLU; visualization of activations and connection strengths
- Adjustable canvas with input centering for 8x8, 28x28 (MNIST) and other square sizes
- Drawing tools:
  - Brush size with square mapping (1→1x1, 2→3x3, 3→4x4, ...)
  - Full-intensity toggle to stamp maximum intensity (1.0)
  - Clear canvas button
- Bundled sample models and UI to paste your own C header

## Tech Stack

- React + TypeScript + Vite
- Tailwind CSS + shadcn/ui
- Static build suitable for GitHub Pages or any static host

## Quick Start (Local)

1. Clone and install:

```bash
git clone https://github.com/ankitmawle/neural-network-visualizer.git
cd neural-network-visualizer/webcode
npm install
```

2. Run the dev server:

```bash
npm run dev
```

Open the shown localhost URL (e.g., `http://localhost:5173/`).

3. Build a production bundle:

```bash
npm run build
```

This creates a `dist/` folder you can self-host on any static server (Nginx, Netlify, S3, etc.).

## Using the Visualizer

### Draw Input
- Use the brush size slider below the canvas to control the footprint:
  - 1 → 1x1, 2 → 3x3, 3 → 4x4, and so on
- Enable "Full intensity (1.0)" to stamp maximum intensity pixels
- Click "Clear" to reset the canvas

### Load a Model
- JSON weights: click "Load Weights" and select a JSON file
- Quantized C header: paste the file contents in the textarea (supports layers L1..L3 or L4)
- If class labels are absent, they default to numeric strings based on output size

### Visualization
- Adjust the Connection Threshold slider to filter weaker connections
- Toggle input centering if desired
- Layer sizes automatically adapt to the loaded model

## Self-Deployment

This repository includes an easy GitHub Pages flow. It builds a variant configured for Pages and publishes it to the `gh-pages` branch.

1. Make sure your changes are committed and pushed to GitHub.
2. Build and publish:

```bash
npm run deploy
```

This runs the following scripts under the hood:
- `build:github` – builds with GitHub Pages mode to `dist-github/`
- `predeploy` – ensures Pages-friendly output (including `.nojekyll`)
- `deploy` – pushes the build to the `gh-pages` branch

3. In your GitHub repository:
   - Settings → Pages → Source → select `gh-pages` branch → Save

Your site will be live at `https://<your-username>.github.io/<your-repo>/`.

### Manual Self-Host

If you prefer your own hosting:

```bash
npm run build
# serve the 'dist/' directory using any static server
```

No server-side code is required.

## Notes & Tips

- Valid C headers include layer defines such as `L1_incoming_weights`, `L1_outgoing_weights`, `L1_bitperweight` and a `const uint32_t L*_weights[]` array.
- The parser infers whether the model has 2 or 3 hidden layers from the number of layers.
- Input grid adapts: 64→8x8, 784→28x28, or any perfect square.

## Credits

- App by [Ankit Mawle](https://www.linkedin.com/in/ankitmawle/)
- UI stack based on a Vite + TS + Tailwind + shadcn/ui starter
