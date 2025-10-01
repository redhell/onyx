import { useFormContext } from "@/components/context/FormContext";
import { SettingsContext } from "@/components/settings/SettingsProvider";
import { credentialTemplates } from "@/lib/connectors/credentials";
import { useUser } from "@/components/user/UserProvider";
import { useContext } from "react";
import { User } from "@/lib/types";
import Text from "@/refresh-components/Text";
import NavigationTab from "@/refresh-components/buttons/NavigationTab";
import SvgSettings from "@/icons/settings";
import { LogoComponent } from "@/components/logo/FixedLogo";

function BackButton({
  isAdmin,
  isCurator,
  user,
}: {
  isAdmin: boolean;
  isCurator: boolean;
  user: User | null;
}) {
  const buttonText = isAdmin ? "Admin Page" : "Curator Page";

  if (!isAdmin && !isCurator) {
    console.error(
      `User is neither admin nor curator, defaulting to curator view. Found user:\n ${JSON.stringify(
        user,
        null,
        2
      )}`
    );
  }

  return (
    <NavigationTab
      icon={SvgSettings}
      className="bg-background-tint-00"
      href="/admin/add-connector"
    >
      {buttonText}
    </NavigationTab>
  );
}

export default function Sidebar() {
  const { formStep, setFormStep, connector, allowAdvanced, allowCreate } =
    useFormContext();
  const combinedSettings = useContext(SettingsContext);
  const { isCurator, isAdmin, user } = useUser();
  if (!combinedSettings) {
    return null;
  }
  const enterpriseSettings = combinedSettings.enterpriseSettings;
  const noCredential = credentialTemplates[connector] == null;

  const settingSteps = [
    ...(!noCredential ? ["Credential"] : []),
    "Connector",
    ...(connector == "file" ? [] : ["Advanced (optional)"]),
  ];

  return (
    <div className="flex flex-col h-screen w-[15rem] bg-background-tint-02 py-padding-content px-padding-button gap-padding-content">
      <div className="flex flex-col items-start justify-center">
        <LogoComponent enterpriseSettings={enterpriseSettings} />
      </div>

      <BackButton isAdmin={isAdmin} isCurator={isCurator} user={user} />

      <div className="h-full flex">
        <div className="mx-auto w-full max-w-2xl px-4 py-8">
          <div className="relative">
            {connector != "file" && (
              <div className="absolute h-[85%] left-[6px] top-[8px] bottom-0 w-0.5 bg-background-tint-04"></div>
            )}
            {settingSteps.map((step, index) => {
              const allowed =
                (step == "Connector" && allowCreate) ||
                (step == "Advanced (optional)" && allowAdvanced) ||
                index <= formStep;

              return (
                <div
                  key={index}
                  className={`flex items-center mb-6 relative ${
                    !allowed ? "cursor-not-allowed" : "cursor-pointer"
                  }`}
                  onClick={() => {
                    if (allowed) {
                      setFormStep(index - (noCredential ? 1 : 0));
                    }
                  }}
                >
                  <div className="flex-shrink-0 mr-4 z-10">
                    <div
                      className={`rounded-full h-3.5 w-3.5 flex items-center justify-center ${
                        allowed ? "bg-blue-500" : "bg-background-tint-04"
                      }`}
                    >
                      {formStep === index && (
                        <div className="h-2 w-2 rounded-full bg-white"></div>
                      )}
                    </div>
                  </div>
                  <Text text04={index <= formStep} text02={index > formStep}>
                    {step}
                  </Text>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
