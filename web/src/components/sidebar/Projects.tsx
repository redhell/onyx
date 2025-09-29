"use client";

import React, { useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  Project,
  useProjectsContext,
} from "@/app/chat/projects/ProjectsContext";
import NavigationTab from "@/components-2/buttons/NavigationTab";
import SvgFolder from "@/icons/folder";
import SvgEdit from "@/icons/edit";
import { PopoverMenu } from "../ui/popover";
import SvgTrash from "@/icons/trash";
import ConfirmationModal from "@/components-2/modals/ConfirmationModal";
import Button from "@/components-2/buttons/Button";
import { ChatButton } from "@/sections/sidebar/AppSidebar";
import { buildChatUrl } from "@/app/chat/services/lib";

interface ProjectProps {
  project: Project;
}

function ProjectFolder({ project }: ProjectProps) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [open, setOpen] = useState(false);
  const [deleteConfirmationModalOpen, setDeleteConfirmationModalOpen] =
    useState(false);
  const { currentProjectId, renameProject, deleteProject } =
    useProjectsContext();
  const [isEditing, setIsEditing] = useState(false);
  const [name, setName] = useState(project.name);

  const active = currentProjectId === project.id;

  async function submitRename(renamedValue: string) {
    const newName = renamedValue.trim();
    if (newName === "") return;

    setName(newName);
    setIsEditing(false);
    await renameProject(project.id, newName);
  }

  return (
    <>
      {/* Confirmation Modal (only for deletion) */}
      {deleteConfirmationModalOpen && (
        <ConfirmationModal
          title="Delete Project"
          icon={SvgTrash}
          onClose={() => setDeleteConfirmationModalOpen(false)}
          description="Are you sure you want to delete this project? This action cannot be undone."
        >
          <div className="flex flex-row justify-end items-center gap-spacing-interline">
            <Button
              onClick={() => setDeleteConfirmationModalOpen(false)}
              secondary
            >
              Cancel
            </Button>
            <Button
              danger
              onClick={() => {
                setDeleteConfirmationModalOpen(false);
                deleteProject(project.id);
              }}
            >
              Delete
            </Button>
          </div>
        </ConfirmationModal>
      )}

      {/* Project Folder */}
      <NavigationTab
        icon={SvgFolder}
        active={active}
        onClick={() => router.push(buildChatUrl(searchParams, null, null))}
        popover={
          <PopoverMenu>
            {[
              <NavigationTab icon={SvgEdit} onClick={() => setIsEditing(true)}>
                Rename Project
              </NavigationTab>,
              null,
              <NavigationTab
                icon={SvgTrash}
                onClick={() => setDeleteConfirmationModalOpen(true)}
                danger
              >
                Delete Project
              </NavigationTab>,
            ]}
          </PopoverMenu>
        }
        renaming={isEditing}
        setRenaming={setIsEditing}
        submitRename={submitRename}
      >
        {name}
      </NavigationTab>

      {/* Project Chat-Sessions */}
      {true &&
        project.chat_sessions.map((chatSession) => (
          <ChatButton chatSession={chatSession} />
        ))}
    </>
  );

  // return (
  //   <div className="w-full">
  //     <div
  //       className={`w-full group flex items-center gap-x-1 px-1 rounded-md hover:bg-background-chat-hover ${isSelected ? "bg-background-chat-selected" : ""}`}
  //     >
  //       <button
  //         type="button"
  //         aria-expanded={open}
  //         onClick={() =>
  //           setOpen((v) => {
  //             const next = !v;
  //             onToggle?.(next);
  //             return next;
  //           })
  //         }
  //         onMouseEnter={() => setHoveringIcon(true)}
  //         onMouseLeave={() => setHoveringIcon(false)}
  //         className="cursor-pointer text-base rounded-md p-1"
  //       >
  //         {open || hoveringIcon ? (
  //           <FolderOpen
  //             size={18}
  //             className="flex-none text-text-history-sidebar-button"
  //           />
  //         ) : (
  //           <FolderIcon
  //             size={18}
  //             className="flex-none text-text-history-sidebar-button"
  //           />
  //         )}
  //       </button>
  //       {isEditing ? (
  //         <input
  //           className="w-full text-base bg-transparent outline-none text-black dark:text-[#D4D4D4] py-1 rounded-md border-b border-transparent focus:border-accent-background-hovered"
  //           value={editValue}
  //           onChange={(e) => setEditValue(e.target.value)}
  //           onKeyDown={async (e) => {
  //             if (e.key === "Enter") {
  //               if (!onRename) return;
  //               const nextName = editValue.trim();
  //               if (!nextName || nextName === title) {
  //                 setIsEditing(false);
  //                 setEditValue(title);
  //                 return;
  //               }
  //               try {
  //                 setIsSaving(true);
  //                 await onRename(nextName);
  //               } finally {
  //                 setIsSaving(false);
  //                 setIsEditing(false);
  //               }
  //             } else if (e.key === "Escape") {
  //               setIsEditing(false);
  //               setEditValue(title);
  //             }
  //           }}
  //           autoFocus
  //         />
  //       ) : (
  //         <button
  //           type="button"
  //           onClick={() => {
  //             setOpen((v) => {
  //               const next = !v;
  //               onToggle?.(next);
  //               return next;
  //             });
  //             onNameClick?.();
  //           }}
  //           className="w-full text-left text-base text-black dark:text-[#D4D4D4] py-1  rounded-md"
  //         >
  //           <span className="truncate">{title}</span>
  //         </button>
  //       )}
  //       <div
  //         className={`ml-2 flex items-center gap-x-1 transition-opacity ${isEditing ? "opacity-100" : "opacity-0 group-hover:opacity-100"}`}
  //       >
  //         {isEditing ? (
  //           <>
  //             <button
  //               type="button"
  //               aria-label="Save name"
  //               disabled={isSaving}
  //               onClick={async (e) => {
  //                 e.stopPropagation();
  //                 if (!onRename) return;
  //                 const nextName = editValue.trim();
  //                 if (!nextName || nextName === title) {
  //                   setIsEditing(false);
  //                   setEditValue(title);
  //                   return;
  //                 }
  //                 try {
  //                   setIsSaving(true);
  //                   await onRename(nextName);
  //                 } finally {
  //                   setIsSaving(false);
  //                   setIsEditing(false);
  //                 }
  //               }}
  //               className="p-1 rounded hover:bg-accent-background-hovered text-green-600 disabled:opacity-50"
  //             >
  //               <Check size={16} />
  //             </button>
  //             <button
  //               type="button"
  //               aria-label="Cancel rename"
  //               onClick={(e) => {
  //                 e.stopPropagation();
  //                 setIsEditing(false);
  //                 setEditValue(title);
  //               }}
  //               className="p-1 rounded hover:bg-accent-background-hovered text-red-600"
  //             >
  //               <X size={16} />
  //             </button>
  //           </>
  //         ) : (
  //           <>
  //             <button
  //               type="button"
  //               aria-label="Rename project"
  //               onClick={(e) => {
  //                 e.stopPropagation();
  //                 setIsEditing(true);
  //                 setEditValue(title);
  //               }}
  //               className="p-1 rounded hover:bg-accent-background-hovered text-text-history-sidebar-button"
  //             >
  //               <Pencil size={16} />
  //             </button>
  //             {onDeleteClick && (
  //               <button
  //                 type="button"
  //                 aria-label="Delete project"
  //                 onClick={(e) => {
  //                   e.stopPropagation();
  //                   onDeleteClick();
  //                 }}
  //                 className="p-1 rounded hover:bg-accent-background-hovered text-text-history-sidebar-button"
  //               >
  //                 <Trash2 size={16} />
  //               </button>
  //             )}
  //           </>
  //         )}
  //       </div>
  //     </div>

  //     <div
  //       className={`grid transition-[grid-template-rows,opacity] duration-300 ease-out ${
  //         open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
  //       }`}
  //     >
  //       <div className="overflow-hidden">
  //         <div className="pl-6 pr-2 py-1 space-y-1">{children}</div>
  //       </div>
  //     </div>
  //   </div>
  // );
}

export default function Projects() {
  const { projects } = useProjectsContext();
  return (
    <>
      {projects.map((project) => (
        <ProjectFolder key={project.id} project={project} />
      ))}
    </>
  );
}
