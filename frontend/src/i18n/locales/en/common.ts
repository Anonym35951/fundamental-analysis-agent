import type { MatchShape } from "../../types";
import type { Dictionary } from "../de";

export const common: MatchShape<Dictionary["common"]> = {
  toast: {
    dismissAriaLabel: "Dismiss notification",
  },
  modal: {
    closeAriaLabel: "Close",
  },
  errorBoundary: {
    title: "Something went wrong.",
    description:
      "This view could not be loaded. Please reload the page. If the problem persists, contact support.",
    reloadButton: "Reload page",
  },
  cookieConsent: {
    brandStrong: "Cloudflare Web Analytics",
    mobileSuffix: "— cookie-free, no PII.",
    mobileLinkLabel: "Details",
    desktopPrefix: "We use",
    desktopMiddle:
      "— privacy-friendly, without cookies and without personal data. Details in our",
    desktopLinkLabel: "cookie notice",
    rejectButton: "Decline",
    acceptButton: "Accept",
  },
};
