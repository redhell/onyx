import SvgFileText from "@/icons/file-text";
import { getFileExtension } from "../files_utils";
import Truncated from "@/refresh-components/Truncated";
import Text from "@/refresh-components/Text";

export function DocumentPreview({
  fileName,
  maxWidth,
  alignBubble,
  open,
}: {
  fileName: string;
  open?: () => void;
  maxWidth?: string;
  alignBubble?: boolean;
}) {
  const typeLabel = getFileExtension(fileName);

  return (
    <div
      className={`relative group flex items-center gap-3 border border-border rounded-xl bg-background-tint-00 px-3 py-1 shadow-sm h-14 w-52 ${
        open ? "cursor-pointer hover:bg-accent-background" : ""
      }`}
      onClick={() => {
        if (open) {
          open();
        }
      }}
    >
      <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-transparent">
        <div className="bg-background-tint-01 p-2 rounded-lg">
          <SvgFileText className="h-5 w-5 stroke-text-02" />
        </div>
      </div>
      <div className="flex flex-col overflow-hidden">
        <Truncated text04 secondaryAction>
          {fileName}
        </Truncated>
        <Text text03 secondaryBody>
          {typeLabel}
        </Text>
      </div>
    </div>
  );
}

export function InputDocumentPreview({
  fileName,
  maxWidth,
  alignBubble,
}: {
  fileName: string;
  maxWidth?: string;
  alignBubble?: boolean;
}) {
  const typeLabel = getFileExtension(fileName);

  return (
    <div
      className={`relative group flex items-center gap-3 border border-border rounded-xl bg-accent-background px-3 py-1 shadow-sm h-14 w-52`}
    >
      <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-transparent">
        <div className="bg-accent-background p-2 rounded-lg shadow-sm">
          <SvgFileText className="h-5 w-5 stroke-text-02" />
        </div>
      </div>
      <div className="flex flex-col overflow-hidden">
        <span className="text-onyx-medium text-sm truncate" title={fileName}>
          {fileName}
        </span>
        <span className="text-onyx-muted text-xs truncate">{typeLabel}</span>
      </div>
    </div>
  );
}
