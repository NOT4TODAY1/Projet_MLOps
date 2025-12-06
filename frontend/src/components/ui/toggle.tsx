import * as React from "react"
import { cn } from "@/lib/utils"

export interface ToggleProps {
  id?: string
  label?: string
  checked: boolean
  onChange: (checked: boolean) => void
  onBlur?: () => void
  disabled?: boolean
  className?: string
  trueLabel?: string
  falseLabel?: string
}

const Toggle = React.forwardRef<HTMLButtonElement, ToggleProps>(
  ({ id, label, checked, onChange, onBlur, disabled, className, trueLabel = "Yes", falseLabel = "No", ...props }, ref) => {
    return (
      <div className={cn("flex flex-col space-y-2", className)}>
        {label && (
          <label htmlFor={id} className="text-white/80 text-sm">
            {label}
          </label>
        )}
        <div className="flex items-center gap-3">
          <button
            type="button"
            id={id}
            ref={ref}
            role="switch"
            aria-checked={checked}
            disabled={disabled}
            onClick={() => {
              if (!disabled) {
                onChange(!checked)
              }
            }}
            onBlur={onBlur}
            className={cn(
              "relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/20 focus-visible:ring-offset-2 focus-visible:ring-offset-neutral-950 disabled:cursor-not-allowed disabled:opacity-50",
              checked ? "bg-white" : "bg-white/20"
            )}
            {...props}
          >
            <span
              className={cn(
                "inline-block h-4 w-4 transform rounded-full bg-neutral-950 transition-transform",
                checked ? "translate-x-6" : "translate-x-1"
              )}
            />
          </button>
          <span className={cn("text-sm font-medium transition-colors", checked ? "text-white" : "text-white/60")}>
            {checked ? trueLabel : falseLabel}
          </span>
        </div>
      </div>
    )
  }
)
Toggle.displayName = "Toggle"

export { Toggle }

