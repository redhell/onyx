import { useMemo, useState } from "react";
import { FiDownload } from "react-icons/fi";
import { FullImageModal } from "./FullImageModal";
import { buildImgUrl } from "./utils";

type InMessageImageProps = {
  fileId: string;
  width?: number | null;
  height?: number | null;
};

export function InMessageImage({ fileId, width, height }: InMessageImageProps) {
  const [fullImageShowing, setFullImageShowing] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [naturalDimensions, setNaturalDimensions] = useState<{
    width: number;
    height: number;
  } | null>(null);

  const handleDownload = async (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent opening the full image modal

    try {
      const response = await fetch(buildImgUrl(fileId));
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `image-${fileId}.png`; // You can adjust the filename/extension as needed
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Failed to download image:", error);
    }
  };

  const resolvedDimensions = useMemo(() => {
    if (width && height) {
      return { width, height };
    }
    if (naturalDimensions) {
      return naturalDimensions;
    }
    return null;
  }, [width, height, naturalDimensions]);

  const aspectRatio = resolvedDimensions
    ? `${resolvedDimensions.width} / ${resolvedDimensions.height}`
    : "1 / 1";

  return (
    <>
      <FullImageModal
        fileId={fileId}
        open={fullImageShowing}
        onOpenChange={(open) => setFullImageShowing(open)}
      />

      <div className="relative w-full max-w-96 group" style={{ aspectRatio }}>
        {!imageLoaded && (
          <div className="absolute inset-0 bg-background-200 animate-pulse rounded-lg" />
        )}

        <img
          width={1200}
          height={1200}
          alt="Chat Message Image"
          onLoad={(event) => {
            setImageLoaded(true);
            if (!width || !height) {
              const { naturalWidth, naturalHeight } = event.currentTarget;
              if (naturalWidth && naturalHeight) {
                setNaturalDimensions({
                  width: naturalWidth,
                  height: naturalHeight,
                });
              }
            }
          }}
          className={`
            object-contain 
            object-left 
            overflow-hidden 
            rounded-lg 
            w-full 
            h-full 
            transition-opacity 
            duration-300 
            cursor-pointer
            ${imageLoaded ? "opacity-100" : "opacity-0"}
          `}
          onClick={() => setFullImageShowing(true)}
          src={buildImgUrl(fileId)}
          loading="lazy"
        />

        {/* Download button - appears on hover */}
        <button
          onClick={handleDownload}
          className="
            absolute 
            bottom-2 
            right-2 
            p-2 
            bg-black/80 
            hover:bg-black 
            text-white 
            dark:text-black
            dark:hover:bg-white
            rounded-lg
            opacity-0 
            group-hover:opacity-100 
            transition-all 
            duration-200 
            z-10 
            shadow-lg 
            hover:shadow-xl
          "
          title="Download"
        >
          <FiDownload className="w-5 h-5" />
        </button>
      </div>
    </>
  );
}
