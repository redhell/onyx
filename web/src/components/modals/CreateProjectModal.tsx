"use client";

import { useRef } from "react";
import Button from "@/components-2/buttons/Button";
import SvgFolderPlus from "@/icons/folder-plus";
import Modal from "@/components-2/modals/Modal";
import { ModalIds, useModal } from "@/components-2/context/ModalContext";
import { useProjectsContext } from "@/app/chat/projects/ProjectsContext";
import { useKeyPress } from "@/hooks/useKeyPress";
import FieldInput from "@/components-2/FieldInput";

export default function CreateProjectModal() {
  const { createProject } = useProjectsContext();
  const { toggleModal } = useModal();
  const fieldInputRef = useRef<HTMLInputElement>(null);

  async function handleSubmit() {
    if (!fieldInputRef.current) return;
    const name = fieldInputRef.current.value.trim();
    if (!name) return;

    try {
      createProject(name);
    } catch (e) {
      console.error(`Failed to create the project ${name}`);
    }

    toggleModal(ModalIds.CreateProjectModal, false);
  }

  useKeyPress(handleSubmit, "Enter");

  return (
    <Modal
      id={ModalIds.CreateProjectModal}
      icon={SvgFolderPlus}
      title="Create New Project"
      description="Use projects to organize your files and chats in one place, and add custom instructions for ongoing work."
      xs
    >
      <div className="flex flex-col p-spacing-paragraph bg-background-tint-01">
        <FieldInput
          label="Project Name"
          placeholder="What are you working on?"
          ref={fieldInputRef}
        />
      </div>
      <div className="flex flex-row justify-end gap-spacing-interline p-spacing-paragraph">
        <Button
          secondary
          onClick={() => toggleModal(ModalIds.CreateProjectModal, false)}
        >
          Cancel
        </Button>
        <Button onClick={handleSubmit}>Create Project</Button>
      </div>
    </Modal>
  );
}
