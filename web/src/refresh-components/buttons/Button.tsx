"use client";

import React from "react";
import Text from "@/refresh-components/Text";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { SvgProps } from "@/icons";

const variantClasses = (active: boolean | undefined) =>
  ({
    primary: {
      main: [
        active ? "bg-theme-primary-06" : "bg-theme-primary-05",
        "hover:bg-theme-primary-04",
      ],
      secondary: [
        active ? "bg-background-tint-00" : "bg-background-tint-01",
        "hover:bg-background-tint-02",
        "border",
      ],
      tertiary: [],
      disabled: [],
    },
    action: {
      main: [],
      secondary: [],
      tertiary: [],
      disabled: [],
    },
    danger: {
      main: [
        active ? "bg-action-danger-06" : "bg-action-danger-05",
        "hover:bg-action-danger-04",
      ],
      secondary: [],
      tertiary: [],
      disabled: [],
    },
  }) as const;

const textClasses = (active: boolean | undefined) =>
  ({
    primary: {
      main: ["text-text-inverted-05"],
      secondary: [
        active ? "text-text-05" : "text-text-03",
        "group-hover:text-text-04",
      ],
      tertiary: [],
      disabled: ["text-text-01"],
    },
    action: {
      main: ["text-text-inverted-05"],
      secondary: [],
      tertiary: [],
      disabled: [],
    },
    danger: {
      main: ["text-text-05"],
      secondary: [],
      tertiary: [],
      disabled: [],
    },
  }) as const;

interface ButtonProps extends React.HTMLAttributes<HTMLButtonElement> {
  // Button variants:
  primary?: boolean;
  action?: boolean;
  danger?: boolean;

  // Button subvariants:
  main?: boolean;
  secondary?: boolean;
  tertiary?: boolean;
  disabled?: boolean;

  // Button states:
  active?: boolean;

  // Icons:
  leftIcon?: React.FunctionComponent<SvgProps>;
  rightIcon?: React.FunctionComponent<SvgProps>;

  href?: string;
}

export default function Button({
  primary,
  action,
  danger,

  main,
  secondary,
  tertiary,
  disabled,

  active,

  leftIcon: LeftIcon,
  rightIcon: RightIcon,

  href,
  children,
  className,
  ...props
}: ButtonProps) {
  const variant = primary
    ? "primary"
    : action
      ? "action"
      : danger
        ? "danger"
        : "primary";

  const subvariant = main
    ? "main"
    : secondary
      ? "secondary"
      : tertiary
        ? "tertiary"
        : disabled
          ? "disabled"
          : "main";

  const content = (
    <button
      className={cn(
        "p-spacing-interline h-fit rounded-12 group w-fit flex flex-row items-center justify-center gap-spacing-inline",
        variantClasses(active)[variant][subvariant],
        className
      )}
      {...props}
    >
      {LeftIcon && (
        <div className="w-[1rem] h-[1rem] flex flex-col items-center justify-center">
          <LeftIcon className="w-[1rem] h-[1rem] stroke-text-inverted-05" />
        </div>
      )}
      {typeof children === "string" ? (
        <Text className={cn(textClasses(active)[variant][subvariant])}>
          {children}
        </Text>
      ) : (
        children
      )}
      {RightIcon && (
        <div className="w-[1rem] h-[1rem]">
          <RightIcon className="w-[1rem] h-[1rem] stroke-text-inverted-05" />
        </div>
      )}
    </button>
  );

  if (!href) return content;

  return <Link href={href}>{content}</Link>;
}
