# Alzheimer's Disease Classifier - Frontend

A modern React + TypeScript web UI for the Alzheimer's Disease Classification API.

## Features

- **Modern UI**: Built with React, TypeScript, and Tailwind CSS
- **shadcn/ui Components**: Beautiful, accessible component library
- **Form Validation**: Real-time validation with error feedback
- **API Integration**: Seamless connection to FastAPI backend
- **Responsive Design**: Works on desktop and mobile devices

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **shadcn/ui** - Component library
- **Lucide React** - Icons

## Setup Instructions

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file (optional, defaults to `http://localhost:8000`):
```env
VITE_API_URL=http://localhost:8000
```

4. Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   └── ui/          # shadcn/ui components
│   ├── lib/
│   │   └── utils.ts     # Utility functions
│   ├── services/
│   │   └── api.ts       # API service layer
│   ├── App.tsx          # Main application component
│   ├── main.tsx         # Application entry point
│   └── index.css        # Global styles
├── index.html
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── vite.config.ts
```

## API Integration

The frontend connects to the FastAPI backend running on `http://localhost:8000` by default. Make sure the backend is running before using the frontend.

### Available Endpoints

- `POST /predict` - Make a prediction
- `GET /health` - Check API health
- `POST /retrain` - Retrain models

## Component Structure

### shadcn/ui Components

All components are located in `src/components/ui/`:

- `input.tsx` - Base input component
- `input-with-feedback.tsx` - Input with error/helper text
- `button.tsx` - Button component
- `card.tsx` - Card container
- `label.tsx` - Form label

### Why `/components/ui`?

The `/components/ui` folder is the standard location for shadcn/ui components. This structure:
- Keeps UI components organized and separate from business logic
- Follows shadcn/ui conventions
- Makes it easy to add more components via the shadcn CLI
- Maintains consistency with the shadcn ecosystem

## Development

### Adding New shadcn Components

You can add more shadcn components using the CLI:

```bash
npx shadcn-ui@latest add [component-name]
```

Or manually copy components to `src/components/ui/`.

### Styling

The project uses Tailwind CSS with custom CSS variables for theming. Colors and other design tokens are defined in `src/index.css`.

## License

Same as the main project.

