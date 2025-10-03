"use client";

import React, { useRef, useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Files } from "@phosphor-icons/react";
import { FileIcon, Loader2, Eye } from "lucide-react";
import { cn } from "@/lib/utils";
import FilesList from "./FilesList";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ProjectFile } from "../../projects/projectsService";
import LineItem from "@/refresh-components/buttons/LineItem";
import SvgPaperclip from "@/icons/paperclip";
import IconButton from "@/refresh-components/buttons/IconButton";
import MoreHorizontal from "@/icons/more-horizontal";

// Small helper to render an icon + label row
const Row = ({ children }: { children: React.ReactNode }) => (
  <div className="flex items-center gap-2 w-full">{children}</div>
);

interface FilePickerContentsProps {
  recentFiles: ProjectFile[];
  onPickRecent?: (file: ProjectFile) => void;
  onFileClick?: (file: ProjectFile) => void;
  triggerUploadPicker: () => void;
  setShowRecentFiles: (show: boolean) => void;
}

const getFileExtension = (fileName: string): string => {
  const idx = fileName.lastIndexOf(".");
  if (idx === -1) return "";
  const ext = fileName.slice(idx + 1).toLowerCase();
  if (ext === "txt") return "PLAINTEXT";
  return ext.toUpperCase();
};

export function FilePickerContents({
  recentFiles,
  onPickRecent,
  onFileClick,
  triggerUploadPicker,
  setShowRecentFiles,
}: FilePickerContentsProps) {
  return (
    <>
      {recentFiles.length > 0 && (
        <>
          <label className="font-secondary-body text-text-02 mx-2 mt-2 mb-1">
            Recent Files
          </label>

          {recentFiles.slice(0, 3).map((f) => (
            <button
              key={f.id}
              onClick={() =>
                onPickRecent ? onPickRecent(f) : console.log("Picked recent", f)
              }
              className="w-full rounded-lg hover:bg-background-neutral-02 hover:text-neutral-900 dark:hover:text-neutral-50 text-input-text group"
            >
              <div className="flex items-center w-full m-1 mt-2 p-0.5 group">
                <Row>
                  <div className="p-0.5">
                    {String(f.status).toLowerCase() === "processing" ? (
                      <Loader2 className="h-4 w-4 animate-spin text-text-02" />
                    ) : (
                      <FileIcon className="h-4 w-4 text-text-02" />
                    )}
                  </div>
                  <span
                    className="truncate max-w-[160px] text-text-03 font-main-body"
                    title={f.name}
                  >
                    {f.name}
                  </span>

                  <div className="relative flex items-center ml-auto mr-2">
                    <div className="p-0.5 text-text-02 font-secondary-body group-hover:opacity-0 transition-opacity duration-150">
                      {getFileExtension(f.name)}
                    </div>

                    {onFileClick &&
                      String(f.status).toLowerCase() !== "processing" && (
                        <button
                          title="View file"
                          aria-label="View file"
                          className="absolute inset-0 flex items-center justify-center p-0.5 bg-transparent border-0 outline-none cursor-pointer opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity duration-150"
                          onClick={(e) => {
                            e.stopPropagation();
                            e.preventDefault();
                            onFileClick && onFileClick(f);
                          }}
                        >
                          <Eye className="h-4 w-4 stroke-text-03 hover:stroke-text-02" />
                        </button>
                      )}
                  </div>
                </Row>
              </div>
            </button>
          ))}

          {recentFiles.length > 3 && (
            <button
              onClick={() => setShowRecentFiles(true)}
              className="w-full rounded-lg hover:bg-background-neutral-02 hover:text-neutral-900 dark:hover:text-neutral-50"
            >
              <div className="flex items-center w-full m-1 p-1">
                <Row>
                  <div className="p-0.5">
                    <MoreHorizontal className="h-4 w-4 stroke-text-02" />
                  </div>
                  <span className="text-text-03 font-main-body">
                    All Recent Files
                  </span>
                </Row>
              </div>
            </button>
          )}

          <div className="border-b" />
        </>
      )}

      <LineItem
        icon={SvgPaperclip}
        description="Upload a file from your device"
        onClick={triggerUploadPicker}
      >
        Upload Files
      </LineItem>
    </>
  );
}

interface FilePickerProps {
  className?: string;
  onPickRecent?: (file: ProjectFile) => void;
  onFileClick?: (file: ProjectFile) => void;
  recentFiles: ProjectFile[];
  handleUploadChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  trigger?: React.ReactNode;
}

export default function FilePicker({
  className,
  onPickRecent,
  onFileClick,
  recentFiles,
  handleUploadChange,
  trigger,
}: FilePickerProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [showRecentFiles, setShowRecentFiles] = useState(false);
  const [open, setOpen] = useState(false);

  const triggerUploadPicker = () => fileInputRef.current?.click();

  return (
    <div className={cn("relative", className)}>
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        multiple
        onChange={handleUploadChange}
        accept={"*/*"}
      />
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <div className="relative cursor-pointer flex items-center group rounded-lg text-input-text px-0 h-8">
            {trigger}
          </div>
        </PopoverTrigger>
        <PopoverContent
          align="start"
          sideOffset={6}
          className="w-[15.5rem] max-h-[300px] border-transparent"
          side="top"
        >
          <FilePickerContents
            recentFiles={recentFiles}
            onPickRecent={(file) => {
              onPickRecent && onPickRecent(file);
              setOpen(false);
            }}
            onFileClick={(file) => {
              onFileClick && onFileClick(file);
              setOpen(false);
            }}
            triggerUploadPicker={() => {
              triggerUploadPicker();
              setOpen(false);
            }}
            setShowRecentFiles={(show) => {
              setShowRecentFiles(show);
              // Close the small popover when opening the dialog
              if (show) setOpen(false);
            }}
          />
        </PopoverContent>
      </Popover>

      <Dialog open={showRecentFiles} onOpenChange={setShowRecentFiles}>
        <DialogContent
          className="w-full max-w-lg px-6 py-3 sm:px-6 sm:py-4 focus:outline-none focus-visible:outline-none"
          tabIndex={-1}
          onOpenAutoFocus={(e) => {
            // Prevent auto-focus which can interfere with input
            e.preventDefault();
          }}
        >
          <DialogHeader className="px-0 pt-0 pb-2">
            <Files size={32} />
            <DialogTitle>Recent Files</DialogTitle>
          </DialogHeader>
          <FilesList
            recentFiles={recentFiles}
            onPickRecent={onPickRecent}
            onFileClick={onFileClick}
            handleUploadChange={handleUploadChange}
          />
        </DialogContent>
      </Dialog>
    </div>
  );
}
