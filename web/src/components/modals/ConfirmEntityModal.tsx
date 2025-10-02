import { Modal } from "../Modal";
import Button from "@/refresh-components/buttons/Button";

export const ConfirmEntityModal = ({
  onClose,
  onSubmit,
  entityType,
  entityName,
  additionalDetails,
  actionButtonText,
  action,
  includeCancelButton = true,
  variant = "delete",
  removeConfirmationText = false,
}: {
  entityType: string;
  entityName: string;
  onClose: () => void;
  onSubmit: () => void;
  additionalDetails?: string;
  actionButtonText?: string;
  action?: string;
  includeCancelButton?: boolean;
  variant?: "delete" | "action";
  removeConfirmationText?: boolean;
}) => {
  const isDeleteVariant = variant === "delete";
  const defaultButtonText = isDeleteVariant ? "Delete" : "Confirm";
  const buttonText = actionButtonText || defaultButtonText;

  const getActionText = () => {
    if (action) {
      return action;
    }
    return isDeleteVariant ? "delete" : "modify";
  };

  return (
    <Modal width="rounded max-w-md w-full" onOutsideClick={onClose}>
      <>
        <div className="flex mb-4">
          <h2 className="my-auto text-2xl font-bold">
            {buttonText} {entityType}
          </h2>
        </div>
        {!removeConfirmationText && (
          <p className="mb-4">
            Are you sure you want to {getActionText()} <b>{entityName}</b>?
          </p>
        )}
        {additionalDetails && <p className="mb-4">{additionalDetails}</p>}
        <div className="flex justify-end gap-2">
          {includeCancelButton && (
            <Button onClick={onClose} secondary>
              Cancel
            </Button>
          )}
          <Button onClick={onSubmit} danger={isDeleteVariant}>
            {buttonText}
          </Button>
        </div>
      </>
    </Modal>
  );
};
