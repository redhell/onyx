"use client";

import Button from "@/refresh-components/buttons/Button";
import SvgPlusCircle from "@/icons/plus-circle";

interface CreateButtonProps {
  href?: string;
  onClick?: () => void;
  text?: string;
}

export default function CreateButton({
  href,
  onClick,
  text,
}: CreateButtonProps) {
  return (
    <Button secondary onClick={onClick} leftIcon={SvgPlusCircle} href={href}>
      {text || "Create"}
    </Button>
  );
}
