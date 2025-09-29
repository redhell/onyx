"use client";

import React, { useCallback, useState, memo, useMemo, useRef } from "react";
import { useSettingsContext } from "@/components/settings/SettingsProvider";
import { OnyxLogoTypeIcon, OnyxIcon } from "@/components/icons/icons";
import { MinimalPersonaSnapshot } from "@/app/admin/assistants/interfaces";
import Text from "@/components-2/Text";
import { DragEndEvent } from "@dnd-kit/core";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import SvgSidebar from "@/icons/sidebar";
import SvgEditBig from "@/icons/edit-big";
import SvgMoreHorizontal from "@/icons/more-horizontal";
import SvgLightbulbSimple from "@/icons/lightbulb-simple";
import Settings from "@/sections/sidebar/Settings";
import { SidebarSection } from "@/sections/sidebar/components";
import NavigationTab from "@/components-2/buttons/NavigationTab";
import AgentsModal from "@/sections/AgentsModal";
import { useChatContext } from "@/components-2/context/ChatContext";
import SvgBubbleText from "@/icons/bubble-text";
import {
  deleteChatSession,
  renameChatSession,
  useAppRouter,
} from "@/app/chat/services/lib";
import { useAgentsContext } from "@/components-2/context/AgentsContext";
import { useAppSidebarContext } from "@/components-2/context/AppSidebarContext";
import { ModalIds, useModal } from "@/components-2/context/ModalContext";
import { ChatSession } from "@/app/chat/interfaces";
import ConfirmationModal from "@/components-2/modals/ConfirmationModal";
import SvgTrash from "@/icons/trash";
import SvgShare from "@/icons/share";
import SvgEdit from "@/icons/edit";
import Button from "@/components-2/buttons/Button";
import SvgPin from "@/icons/pin";
import { cn, noProp } from "@/lib/utils";
import { PopoverMenu } from "@/components/ui/popover";
import IconButton from "@/components-2/buttons/IconButton";
import SvgFolderPlus from "@/icons/folder-plus";
import SvgOnyxOctagon from "@/icons/onyx-octagon";
import Projects from "@/components/sidebar/Projects";
import { useProjectsContext } from "@/app/chat/projects/ProjectsContext";
import { useSearchParams } from "next/navigation";
import CreateProjectModal from "@/components/modals/CreateProjectModal";

// Visible-agents = pinned-agents + current-agent (if current-agent not in pinned-agents)
// OR Visible-agents = pinned-agents (if current-agent in pinned-agents)
function buildVisibleAgents(
  pinnedAgents: MinimalPersonaSnapshot[],
  currentAgent: MinimalPersonaSnapshot | null
): [MinimalPersonaSnapshot[], boolean] {
  if (!currentAgent) return [pinnedAgents, false];
  const currentAgentIsPinned = pinnedAgents.some(
    (pinnedAgent) => pinnedAgent.id === currentAgent.id
  );
  const visibleAgents = currentAgentIsPinned
    ? pinnedAgents
    : [...pinnedAgents, currentAgent];
  return [visibleAgents, currentAgentIsPinned];
}

interface ChatButtonProps {
  chatSession: ChatSession;
}

function ChatButtonInner({ chatSession }: ChatButtonProps) {
  const route = useAppRouter();

  const [name, setName] = useState(chatSession.name);
  const [renaming, setRenaming] = useState(false);
  const [deleteConfirmationModalOpen, setDeleteConfirmationModalOpen] =
    useState(false);
  const { refreshChatSessions, currentChat } = useChatContext();

  async function submitRename(renamingValue: string) {
    const newName = renamingValue.trim();
    if (newName === "" || newName === chatSession.name) return;

    setName(newName);
    try {
      await renameChatSession(chatSession.id, newName);
      await refreshChatSessions();
    } catch (error) {
      console.error("Failed to rename chat:", error);
    }
  }

  async function handleChatDelete() {
    try {
      await deleteChatSession(chatSession.id);
      await refreshChatSessions();
    } catch (error) {
      console.error("Failed to delete chat:", error);
    }
  }

  return (
    <>
      {deleteConfirmationModalOpen && (
        <ConfirmationModal
          title="Delete Chat"
          icon={SvgTrash}
          description="Are you sure you want to delete this chat? This action cannot be undone."
          onClose={() => setDeleteConfirmationModalOpen(false)}
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
                handleChatDelete();
              }}
            >
              Delete
            </Button>
          </div>
        </ConfirmationModal>
      )}

      <NavigationTab
        icon={SvgBubbleText}
        onClick={() => route({ chatSessionId: chatSession.id })}
        active={currentChat?.id === chatSession.id}
        popover={
          <PopoverMenu>
            {[
              <NavigationTab key="share" icon={SvgShare} onClick={noProp()}>
                Share
              </NavigationTab>,
              <NavigationTab
                key="rename"
                icon={SvgEdit}
                onClick={noProp(() => setRenaming(true))}
              >
                Rename
              </NavigationTab>,
              null,
              <NavigationTab
                key="delete"
                icon={SvgTrash}
                onClick={noProp(() => setDeleteConfirmationModalOpen(true))}
                danger
              >
                Delete
              </NavigationTab>,
            ]}
          </PopoverMenu>
        }
        renaming={renaming}
        setRenaming={setRenaming}
        submitRename={submitRename}
      >
        {name}
      </NavigationTab>
    </>
  );
}

export const ChatButton = memo(ChatButtonInner);

interface AgentsButtonProps {
  visibleAgent: MinimalPersonaSnapshot;
}

function AgentsButtonInner({ visibleAgent }: AgentsButtonProps) {
  const route = useAppRouter();
  const { currentAgent, pinnedAgents, togglePinnedAgent } = useAgentsContext();
  const pinned = pinnedAgents.some(
    (pinnedAgent) => pinnedAgent.id === visibleAgent.id
  );

  return (
    <SortableItem id={visibleAgent.id}>
      <div className="flex flex-col w-full h-full">
        <NavigationTab
          key={visibleAgent.id}
          icon={SvgLightbulbSimple}
          onClick={() => route({ agentId: visibleAgent.id })}
          active={currentAgent?.id === visibleAgent.id}
          popover={
            <PopoverMenu>
              {[
                <NavigationTab
                  key="pin-unpin-chat"
                  icon={SvgPin}
                  onClick={noProp(() =>
                    togglePinnedAgent(visibleAgent, !pinned)
                  )}
                >
                  {pinned ? "Unpin chat" : "Pin chat"}
                </NavigationTab>,
              ]}
            </PopoverMenu>
          }
          highlight
        >
          {visibleAgent.name}
        </NavigationTab>
      </div>
    </SortableItem>
  );
}

const AgentsButton = memo(AgentsButtonInner);

interface SortableItemProps {
  id: number;
  children?: React.ReactNode;
}

function SortableItem({ id, children }: SortableItemProps) {
  const { attributes, listeners, setNodeRef, transform, isDragging } =
    useSortable({ id });

  return (
    <div
      ref={setNodeRef}
      style={{
        transform: CSS.Transform.toString(transform),
        ...(isDragging && { zIndex: 1000, position: "relative" as const }),
      }}
      {...attributes}
      {...listeners}
      className="flex items-center group"
    >
      {children}
    </div>
  );
}

function AppSidebarInner() {
  const route = useAppRouter();
  const searchParams = useSearchParams();
  const { pinnedAgents, setPinnedAgents, currentAgent } = useAgentsContext();
  const { folded, setFolded } = useAppSidebarContext();
  const { toggleModal } = useModal();
  const { chatSessions } = useChatContext();
  const combinedSettings = useSettingsContext();

  const [visibleAgents, currentAgentIsPinned] = useMemo(
    () => buildVisibleAgents(pinnedAgents, currentAgent),
    [pinnedAgents, currentAgent]
  );
  const visibleAgentIds = useMemo(
    () => visibleAgents.map((agent) => agent.id),
    [visibleAgents]
  );

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8,
      },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const { active, over } = event;
      if (!over) return;
      if (active.id === over.id) return;

      setPinnedAgents((prev) => {
        const activeIndex = visibleAgentIds.findIndex(
          (agentId) => agentId === active.id
        );
        const overIndex = visibleAgentIds.findIndex(
          (agentId) => agentId === over.id
        );

        if (currentAgent && !currentAgentIsPinned) {
          // This is the case in which the user is dragging the UNPINNED agent and moving it to somewhere else in the list.
          // This is an indication that we WANT to pin this agent!
          if (activeIndex === visibleAgentIds.length - 1) {
            const prevWithVisible = [...prev, currentAgent];
            return arrayMove(prevWithVisible, activeIndex, overIndex);
          }
        }

        return arrayMove(prev, activeIndex, overIndex);
      });
    },
    [visibleAgentIds, setPinnedAgents, currentAgent, currentAgentIsPinned]
  );

  const isHistoryEmpty = useMemo(
    () => !chatSessions || chatSessions.length === 0,
    [chatSessions]
  );

  if (!combinedSettings) {
    return null;
  }

  return (
    <>
      <AgentsModal />
      <CreateProjectModal />

      <div
        className={cn(
          "h-full flex flex-col bg-background-tint-02 py-padding-content flex-shrink-0 px-padding-button justify-between group/AppSidebar",
          folded ? "w-[4rem]" : "w-[15rem]"
        )}
      >
        {/* Top */}
        <div className="flex h-full flex-col gap-padding-content">
          <div
            className={cn(
              "flex flex-row items-center px-spacing-interline py-spacing-inline flex-shrink-0",
              folded ? "justify-center" : "justify-between"
            )}
          >
            {folded ? (
              <div className="h-[2rem] flex flex-col justify-center items-center">
                <>
                  <IconButton
                    icon={SvgSidebar}
                    tertiary
                    onClick={() => setFolded(false)}
                    className="hidden group-hover/AppSidebar:flex"
                  />
                  <OnyxIcon
                    size={24}
                    className="visible group-hover/AppSidebar:hidden"
                  />
                </>
              </div>
            ) : (
              <>
                <OnyxLogoTypeIcon size={88} />
                <IconButton
                  icon={SvgSidebar}
                  tertiary
                  onClick={() => setFolded(true)}
                />
              </>
            )}
          </div>

          <div className="flex flex-col gap-spacing-interline">
            <NavigationTab
              icon={SvgEditBig}
              className="!w-full"
              folded={folded}
              onClick={() => route({})}
              active={searchParams.size === 0}
              tooltip
            >
              New Session
            </NavigationTab>

            {folded && (
              <>
                <NavigationTab
                  icon={SvgOnyxOctagon}
                  folded
                  tooltip
                  onClick={() => toggleModal(ModalIds.AgentsModal, true)}
                >
                  Agents
                </NavigationTab>
                <NavigationTab
                  icon={SvgFolderPlus}
                  folded
                  tooltip
                  onClick={() => toggleModal(ModalIds.CreateProjectModal, true)}
                >
                  New Project
                </NavigationTab>
              </>
            )}
          </div>

          {!folded && (
            <div className="flex flex-col gap-padding-content flex-1 overflow-y-scroll">
              {/* Agents */}
              <SidebarSection title="Agents">
                <DndContext
                  sensors={sensors}
                  collisionDetection={closestCenter}
                  onDragEnd={handleDragEnd}
                >
                  <SortableContext
                    items={visibleAgentIds}
                    strategy={verticalListSortingStrategy}
                  >
                    {visibleAgents.map((visibleAgent) => (
                      <AgentsButton
                        key={visibleAgent.id}
                        visibleAgent={visibleAgent}
                      />
                    ))}
                  </SortableContext>
                </DndContext>
                <NavigationTab
                  icon={SvgMoreHorizontal}
                  onClick={() => toggleModal(ModalIds.AgentsModal, true)}
                  lowlight
                >
                  More Agents
                </NavigationTab>
              </SidebarSection>

              <SidebarSection title="Projects">
                <Projects />
              </SidebarSection>

              {/* Recents */}
              <SidebarSection title="Recents">
                {isHistoryEmpty ? (
                  <Text text01 className="px-padding-button">
                    Try sending a message! Your chat history will appear here.
                  </Text>
                ) : (
                  chatSessions.map((chatSession) => (
                    <ChatButton
                      key={chatSession.id}
                      chatSession={chatSession}
                    />
                  ))
                )}
              </SidebarSection>
            </div>
          )}
        </div>

        {/* Bottom */}
        <div className="flex flex-col">
          <Settings folded={folded} />
        </div>
      </div>
    </>
  );
}

const AppSidebar = memo(AppSidebarInner);
AppSidebar.displayName = "AppSidebar";

export default AppSidebar;
