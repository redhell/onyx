import { SettingsContext } from "@/components/settings/SettingsProvider";
import { useContext, ReactNode } from "react";
import NavigationTab from "@/refresh-components/buttons/NavigationTab";
import { LogoComponent } from "@/components/logo/FixedLogo";
import { SvgProps } from "@/icons";

interface StepSidebarProps {
  children: ReactNode;
  buttonName: string;
  buttonIcon: React.FunctionComponent<SvgProps>;
  buttonHref: string;
}

export default function StepSidebar({
  children,
  buttonName,
  buttonIcon,
  buttonHref,
}: StepSidebarProps) {
  const combinedSettings = useContext(SettingsContext);
  if (!combinedSettings) {
    return null;
  }

  const enterpriseSettings = combinedSettings.enterpriseSettings;

  return (
    <div className="flex flex-col h-screen w-[15rem] bg-background-tint-02 py-padding-content px-padding-button gap-padding-content">
      <div className="flex flex-col items-start justify-center">
        <LogoComponent enterpriseSettings={enterpriseSettings} />
      </div>

      <NavigationTab
        icon={buttonIcon}
        className="bg-background-tint-00"
        href={buttonHref}
      >
        {buttonName}
      </NavigationTab>

      <div className="h-full flex">
        <div className="w-full max-w-2xl px-2">{children}</div>
      </div>
    </div>
  );
}
