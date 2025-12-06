import * as React from "react"
import { cn } from "@/lib/utils"
import { Input } from "@/components/ui/input"

export interface InputWithFeedbackProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  errorMessage?: string
  isError?: boolean
  helperText?: string
}

const InputWithFeedback = React.forwardRef<HTMLInputElement, InputWithFeedbackProps>(
  ({ className, errorMessage, helperText, isError, ...props }, ref) => {
    return (
      <div className="relative w-full flex flex-col">
        <Input
          className={cn(
            "w-full",
            className,
            isError && "border-red-500/50 focus-visible:ring-red-500/50"
          )}
          ref={ref}
          {...props}
        />
        
        {isError && errorMessage && (
          <p className="mt-1 text-xs text-red-400 min-h-[1rem]">
            {errorMessage}
          </p>
        )}

        {!isError && helperText && (
          <p className="mt-1 text-xs text-white/50 min-h-[1rem]">
            {helperText}
          </p>
        )}
      </div>
    )
  }
)
InputWithFeedback.displayName = "InputWithFeedback"

export { InputWithFeedback }

