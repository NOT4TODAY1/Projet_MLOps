# Setup Instructions

## Quick Start

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser:**
   Navigate to `http://localhost:3000`

## Prerequisites

- **Node.js**: Version 18 or higher
- **npm/yarn/pnpm**: Package manager
- **Backend API**: The FastAPI backend should be running on `http://localhost:8000`

## Project Structure Explanation

### Why `/components/ui`?

The `/components/ui` folder is the standard location for shadcn/ui components. This structure is important because:

1. **Convention**: Follows shadcn/ui best practices
2. **Organization**: Separates UI components from business logic components
3. **CLI Compatibility**: The shadcn CLI automatically places components here
4. **Maintainability**: Makes it easy to find and manage UI components
5. **Scalability**: As you add more components, they'll be organized consistently

### Component Dependencies

The project uses:
- **shadcn/ui** components (Input, Button, Card, Label)
- **Tailwind CSS** for styling
- **Lucide React** for icons
- **class-variance-authority** for component variants
- **clsx** and **tailwind-merge** for className utilities

## Configuration Files

- `package.json` - Dependencies and scripts
- `tsconfig.json` - TypeScript configuration
- `tailwind.config.js` - Tailwind CSS configuration
- `vite.config.ts` - Vite build configuration
- `components.json` - shadcn/ui configuration
- `.eslintrc.cjs` - ESLint configuration

## Environment Variables

Create a `.env` file in the `frontend` directory (optional):

```env
VITE_API_URL=http://localhost:8000
```

If not set, it defaults to `http://localhost:8000`.

## Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Troubleshooting

### Port Already in Use
If port 3000 is already in use, Vite will automatically try the next available port.

### API Connection Issues
Make sure the FastAPI backend is running on `http://localhost:8000`. You can check the health endpoint:
```bash
curl http://localhost:8000/health
```

### TypeScript Errors
Run the type checker:
```bash
npx tsc --noEmit
```

### Missing Dependencies
If you encounter missing module errors, try:
```bash
rm -rf node_modules package-lock.json
npm install
```

