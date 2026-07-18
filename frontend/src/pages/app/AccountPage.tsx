import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  deleteAccount,
  getCurrentUser,
  invalidateCurrentUserCache,
  updateProfile,
  type CurrentUserResponse,
} from "../../api/auth";
import {
  changePassword,
  cancelSubscription as cancelSubscriptionRequest,
  resumeSubscription as resumeSubscriptionRequest,
  createPortalSession,
} from "../../api/account";
import { clearAuthAndWorkspaceState } from "../../api/client";
import CancelSubscriptionModal from "../../components/account/CancelSubscriptionModal";
import Modal from "../../components/ui/Modal";
import Input from "../../components/ui/Input";
import { theme } from "../../components/ui/theme";
import ParticleBeamBackground from "../../components/landing/ParticleBeamBackground";
import { MIN_AGE, calculateAge } from "../../lib/age";
import { useLocale } from "../../i18n/useLocale";
import type { Locale } from "../../i18n/config";
import { useToast } from "../../components/ui/useToast";

const TODAY_ISO = new Date().toISOString().slice(0, 10);

function AccountPage() {
  const navigate = useNavigate();
  const { locale, setLocale } = useLocale();
  const { showToast } = useToast();
  const [currentUser, setCurrentUser] = useState<CurrentUserResponse | null>(
    null
  );
  const [isLoadingUser, setIsLoadingUser] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");
  const [isSavingLocale, setIsSavingLocale] = useState(false);

  // Optionale Profil-Nachpflege (Benutzername/Vorname/Nachname/Geburtsdatum) -
  // gleiches State-Namensmuster wie die Passwort-Sektion unten.
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [profileErrorMessage, setProfileErrorMessage] = useState("");
  const [profileSuccessMessage, setProfileSuccessMessage] = useState("");
  const [profileUsername, setProfileUsername] = useState("");
  const [profileFirstName, setProfileFirstName] = useState("");
  const [profileLastName, setProfileLastName] = useState("");
  const [profileBirthDate, setProfileBirthDate] = useState("");

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmNewPassword, setConfirmNewPassword] = useState("");
  const [passwordErrorMessage, setPasswordErrorMessage] = useState("");
  const [passwordSuccessMessage, setPasswordSuccessMessage] = useState("");
  const [isChangingPassword, setIsChangingPassword] = useState(false);

  const [subscriptionErrorMessage, setSubscriptionErrorMessage] = useState("");
  const [subscriptionSuccessMessage, setSubscriptionSuccessMessage] =
    useState("");
  const [isCancelingSubscription, setIsCancelingSubscription] = useState(false);
  const [isResumingSubscription, setIsResumingSubscription] = useState(false);
  const [isCancelModalOpen, setIsCancelModalOpen] = useState(false);
  const [isOpeningPortal, setIsOpeningPortal] = useState(false);

  // DSGVO-Konto-Löschung (Hard Delete, Passwort-bestätigt)
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [deletePassword, setDeletePassword] = useState("");
  const [deleteErrorMessage, setDeleteErrorMessage] = useState("");
  const [isDeletingAccount, setIsDeletingAccount] = useState(false);


  useEffect(() => {
    async function loadCurrentUser() {
      try {
        setErrorMessage("");
        const user = await getCurrentUser();
        setCurrentUser(user);
      } catch (error) {
        if (error instanceof Error) {
          setErrorMessage(error.message);
        } else {
          setErrorMessage("Benutzerdaten konnten nicht geladen werden.");
        }
      } finally {
        setIsLoadingUser(false);
      }
    }

    loadCurrentUser();
  }, []);

  // EVOLVING.md § Internationalisierung, I18N-005: optimistisch — die UI
  // wechselt sofort (Context + localStorage via setLocale), das PATCH läuft
  // parallel im Hintergrund. Ein Fehlschlag revertiert die UI-Sprache
  // absichtlich NICHT (kein Gezucke), sondern zeigt nur einen Hinweis, dass
  // die Präferenz nicht im Konto gespeichert werden konnte.
  async function handleSelectLocale(next: Locale) {
    if (next === locale || isSavingLocale) {
      return;
    }

    setLocale(next);
    setIsSavingLocale(true);

    try {
      const updated = await updateProfile({ locale: next });
      setCurrentUser(updated);
    } catch {
      showToast("Sprache konnte nicht im Konto gespeichert werden.", "error");
    } finally {
      setIsSavingLocale(false);
    }
  }

  function handleStartEditProfile() {
    setProfileErrorMessage("");
    setProfileSuccessMessage("");
    setProfileUsername(currentUser?.username ?? "");
    setProfileFirstName(currentUser?.first_name ?? "");
    setProfileLastName(currentUser?.last_name ?? "");
    // Nur aus birth_date vorbefuellen, nicht aus dem Legacy-age-Feld
    // zurueckrechnen (waere nur eine Schaetzung, keine echte Angabe).
    setProfileBirthDate(currentUser?.birth_date ?? "");
    setIsEditingProfile(true);
  }

  async function handleSaveProfile(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (isSavingProfile) {
      return;
    }

    setProfileErrorMessage("");
    setProfileSuccessMessage("");

    const trimmedUsername = profileUsername.trim();
    const trimmedFirstName = profileFirstName.trim();
    const trimmedLastName = profileLastName.trim();

    if (trimmedUsername && !/^[a-zA-Z0-9_.-]{3,50}$/.test(trimmedUsername)) {
      setProfileErrorMessage(
        "Der Benutzername muss 3-50 Zeichen lang sein und darf nur Buchstaben, Zahlen, Punkt, Unterstrich oder Bindestrich enthalten."
      );
      return;
    }

    if (profileBirthDate) {
      if (profileBirthDate > TODAY_ISO) {
        setProfileErrorMessage("Das Geburtsdatum darf nicht in der Zukunft liegen.");
        return;
      }
      if (calculateAge(profileBirthDate) < MIN_AGE) {
        setProfileErrorMessage(`Du musst mindestens ${MIN_AGE} Jahre alt sein.`);
        return;
      }
    }

    setIsSavingProfile(true);

    try {
      const updated = await updateProfile({
        ...(trimmedUsername ? { username: trimmedUsername } : {}),
        ...(trimmedFirstName ? { first_name: trimmedFirstName } : {}),
        ...(trimmedLastName ? { last_name: trimmedLastName } : {}),
        ...(profileBirthDate ? { birth_date: profileBirthDate } : {}),
      });
      setCurrentUser(updated);
      setIsEditingProfile(false);
      setProfileSuccessMessage("Profil aktualisiert.");
    } catch (error) {
      if (error instanceof Error) {
        setProfileErrorMessage(error.message);
      } else {
        setProfileErrorMessage("Profil konnte nicht gespeichert werden.");
      }
    } finally {
      setIsSavingProfile(false);
    }
  }

  const formattedCreatedAt = currentUser?.created_at
    ? new Date(currentUser.created_at).toLocaleDateString("de-DE", {
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
      })
    : "Wird geladen";

  const formattedCurrentPeriodEnd = currentUser?.current_period_end
    ? new Date(currentUser.current_period_end).toLocaleDateString("de-DE", {
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
      })
    : "Kein Enddatum verfügbar";

  const planLabel = currentUser ? getPlanLabel(currentUser.plan) : "Wird geladen";
  const billingStatusLabel = currentUser
    ? getBillingStatusLabel(currentUser.billing_status)
    : "Wird geladen";
  const billingIntervalLabel = currentUser
    ? getBillingIntervalLabel(currentUser.billing_interval)
    : "Wird geladen";

  const isPro = currentUser?.plan === "pro";
  const isSubscriptionCanceling = currentUser?.billing_status === "canceling";
  // 🔽 NUR DIESE ERWEITERUNG IST NEU (oben bei den states ergänzen)
  const isPaymentFailedCanceled =
    currentUser?.billing_status === "payment_failed_canceled";
  const hasStripeCustomer = Boolean(currentUser?.stripe_customer_id);
const canManageSubscriptionPortal =
  hasStripeCustomer && (isPro || isSubscriptionCanceling);
  async function handleDeleteAccount() {
    if (isDeletingAccount) {
      return;
    }

    if (!deletePassword) {
      setDeleteErrorMessage("Bitte gib dein Passwort zur Bestätigung ein.");
      return;
    }

    setDeleteErrorMessage("");
    setIsDeletingAccount(true);

    try {
      await deleteAccount(deletePassword);
      clearAuthAndWorkspaceState();
      // Sticky notice fürs LoginPage, überlebt den Redirect (gleiches Muster
      // wie password_reset_success / registration_verify_pending).
      sessionStorage.setItem("account_deleted_success", "1");
      navigate("/login");
    } catch (error) {
      if (error instanceof Error) {
        if (error.message === "Current password is incorrect") {
          setDeleteErrorMessage("Das Passwort ist nicht korrekt.");
        } else {
          setDeleteErrorMessage(error.message);
        }
      } else {
        setDeleteErrorMessage("Konto konnte nicht gelöscht werden.");
      }
      setIsDeletingAccount(false);
    }
  }

  async function handleChangePassword(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (isChangingPassword) {
      return;
    }

    setPasswordErrorMessage("");
    setPasswordSuccessMessage("");

    if (!currentPassword || !newPassword || !confirmNewPassword) {
      setPasswordErrorMessage("Bitte fülle alle Passwort-Felder aus.");
      return;
    }

    if (newPassword.length < 8) {
      setPasswordErrorMessage(
        "Das neue Passwort muss mindestens 8 Zeichen lang sein."
      );
      return;
    }

    if (newPassword !== confirmNewPassword) {
      setPasswordErrorMessage("Die neuen Passwörter stimmen nicht überein.");
      return;
    }

    if (currentPassword === newPassword) {
      setPasswordErrorMessage(
        "Das neue Passwort sollte sich vom aktuellen Passwort unterscheiden."
      );
      return;
    }

    try {
      setIsChangingPassword(true);

      await changePassword(currentPassword, newPassword);

      setCurrentPassword("");
      setNewPassword("");
      setConfirmNewPassword("");
      setPasswordSuccessMessage(
        "Dein Passwort wurde erfolgreich geändert. Du erhältst eine Bestätigung per E-Mail."
      );
    } catch (error) {
      if (error instanceof Error) {
        setPasswordErrorMessage(error.message);
      } else {
        setPasswordErrorMessage("Passwort konnte nicht geändert werden.");
      }
    } finally {
      setIsChangingPassword(false);
    }
  }

  async function handleCancelSubscription(reason: string) {
    if (isCancelingSubscription || isSubscriptionCanceling) {
      return;
    }

    setSubscriptionErrorMessage("");
    setSubscriptionSuccessMessage("");

    try {
      setIsCancelingSubscription(true);

      await cancelSubscriptionRequest(reason || undefined);

      setSubscriptionSuccessMessage(
        "Dein Abonnement wurde zur Kündigung vorgemerkt und endet zum Ende des aktuellen Abrechnungszeitraums."
      );

      try {
        // EV-112: ohne diesen Invalidate wuerde ein innerhalb der letzten
        // 30s bereits gecachter (jetzt veralteter) Nutzer zurueckgegeben,
        // statt des durch die vorherige Mutation tatsaechlich geaenderten
        // billing_status.
        invalidateCurrentUserCache();
        const refreshedUser = await getCurrentUser();
        setCurrentUser(refreshedUser);
      } catch {
        // UI-Meldung bleibt trotzdem erhalten
      }
    } catch (error) {
      if (error instanceof Error) {
        setSubscriptionErrorMessage(error.message);
      } else {
        setSubscriptionErrorMessage("Abonnement konnte nicht gekündigt werden.");
      }
    } finally {
      setIsCancelingSubscription(false);
      setIsCancelModalOpen(false);
    }
  }

  async function handleResumeSubscription() {
    if (isResumingSubscription || !isSubscriptionCanceling) {
      return;
    }

    setSubscriptionErrorMessage("");
    setSubscriptionSuccessMessage("");

    try {
      setIsResumingSubscription(true);

      await resumeSubscriptionRequest();

      setSubscriptionSuccessMessage(
        "Dein Pro-Abonnement wurde erfolgreich weitergeführt."
      );

      try {
        // EV-112: ohne diesen Invalidate wuerde ein innerhalb der letzten
        // 30s bereits gecachter (jetzt veralteter) Nutzer zurueckgegeben,
        // statt des durch die vorherige Mutation tatsaechlich geaenderten
        // billing_status.
        invalidateCurrentUserCache();
        const refreshedUser = await getCurrentUser();
        setCurrentUser(refreshedUser);
      } catch {
        // UI-Meldung bleibt trotzdem erhalten
      }
    } catch (error) {
      if (error instanceof Error) {
        setSubscriptionErrorMessage(error.message);
      } else {
        setSubscriptionErrorMessage(
          "Abonnement konnte nicht fortgesetzt werden."
        );
      }
    } finally {
      setIsResumingSubscription(false);
    }
  }

  async function handleOpenCustomerPortal() {
    if (isOpeningPortal) {
      return;
    }

    setErrorMessage("");

    try {
      setIsOpeningPortal(true);

      const url = await createPortalSession();
      window.location.href = url;
    } catch (error) {
      if (error instanceof Error) {
        setErrorMessage(error.message);
      } else {
        setErrorMessage("Das Zahlungsportal konnte nicht geöffnet werden.");
      }
    } finally {
      setIsOpeningPortal(false);
    }
  }

  return (
    <>
      <CancelSubscriptionModal
        isOpen={isCancelModalOpen}
        isLoading={isCancelingSubscription}
        onConfirm={handleCancelSubscription}
        onCancel={() => {
          if (!isCancelingSubscription) {
            setIsCancelModalOpen(false);
          }
        }}
      />

      <div

        style={{

          position: "relative",

          display: "flex",

          flexDirection: "column",

          gap: "28px",

          padding: "10px 6px 20px",

        }}

      >

        <ParticleBeamBackground densityMultiplier={1.3} />

        {errorMessage ? (

          <div

            style={{

              position: "relative",

              padding: "14px 16px",

              borderRadius: "14px",

              background: theme.colors.dangerSoft,

              border: `1px solid ${theme.colors.dangerBorder}`,

              color: theme.colors.dangerText,

              fontSize: "0.98rem",

              lineHeight: 1.7,

            }}

          >

            {errorMessage}

          </div>

        ) : null}

        {/* ✅ NEU: PAYMENT FAILED WARNUNG */}

        {isPaymentFailedCanceled ? (

          <div

            style={{

              position: "relative",

              padding: "14px 16px",

              borderRadius: "14px",

              background: theme.colors.dangerSoft,

              border: `1px solid ${theme.colors.dangerBorder}`,

              color: theme.colors.dangerText,

              fontSize: "0.98rem",

              lineHeight: 1.7,

            }}

          >
            Deine letzte Zahlung ist fehlgeschlagen und dein Pro-Zugang wurde beendet. Um Pro wieder zu nutzen, musst du ein neues Abonnement abschließen.
          </div>

        ) : null}

        <section style={heroSection}>
          <div style={heroBadge}>Konto & Mitgliedschaft</div>

          <h1
            style={{
              margin: "0 0 14px 0",
              fontSize: "3rem",
              lineHeight: 1.05,
              letterSpacing: "-0.045em",
              color: theme.colors.textPrimary,
            }}
          >
            Verwalte dein Konto an einem Ort
          </h1>

          <p
            style={{
              margin: 0,
              maxWidth: "860px",
              color: theme.colors.textPrimary,
              fontSize: "1.12rem",
              lineHeight: 1.9,
            }}
          >
            Hier findest du deine wichtigsten Konto- und Abo-Informationen
            gebündelt an einem Ort. So bekommst du schnell einen Überblick über
            deinen aktuellen Zugang und kannst bei Bedarf direkt in den
            Billing-Bereich wechseln.
          </p>

          <div style={heroActionRow}>
            <Link to="/app/billing" style={primaryButton}>
              Billing verwalten
            </Link>

            <Link to="/app/dashboard" style={secondaryButton}>
              Zum Dashboard
            </Link>
          </div>
        </section>

        <section style={contentGrid}>
          <div style={mainCard}>
            <div style={sectionEyebrow}>Kontoinformationen</div>
            <div style={cardTitle}>Dein Profil</div>
            <p style={cardText}>
              Hier siehst du deine aktuellen Kontodaten und den Status deines
              Zugangs. So hast du die wichtigsten Informationen direkt im Blick.
            </p>

            <div style={infoList}>
              <div style={infoRow}>
                <span style={infoLabel}>E-Mail</span>
                <span style={infoValue}>
                  {isLoadingUser ? "Wird geladen..." : currentUser?.email ?? "-"}
                </span>
              </div>

              <div style={infoRow}>
                <span style={infoLabel}>Konto-Status</span>
                {isLoadingUser ? (
                  <span style={infoValueMuted}>Wird geladen...</span>
                ) : (
                  <span style={statusPill}>
                    {currentUser?.is_active ? "Aktiv" : "Inaktiv"}
                  </span>
                )}
              </div>

              <div style={infoRow}>
                <span style={infoLabel}>Mitglied seit</span>
                <span style={infoValueMuted}>
                  {isLoadingUser ? "Wird geladen..." : formattedCreatedAt}
                </span>
              </div>

              <div style={infoRow}>
                <span style={infoLabel}>Billing-Status</span>
                <span style={infoValueMuted}>
                  {isLoadingUser ? "Wird geladen..." : billingStatusLabel}
                </span>
              </div>
            </div>
          </div>

          <div style={sideColumn}>
            <div style={sideCard}>
              <div style={sectionEyebrow}>Aktueller Plan</div>
              <div style={sideCardTitle}>Mitgliedschaft</div>
              <p style={sideCardText}>
                Hier siehst du deinen aktiven Tarif inklusive Status und
                Abrechnungsintervall.
              </p>

              <div style={planBox}>
                <div style={planName}>
                  {isLoadingUser ? "Wird geladen..." : planLabel}
                </div>
                <div style={planHint}>
                  {isLoadingUser
                    ? "Bitte warten..."
                    : isPro
                      ? `Abrechnung: ${billingIntervalLabel}`
                      : "Noch kein Upgrade aktiv"}
                </div>
              </div>

              <div style={membershipInfoList}>
                <div style={membershipInfoRow}>
                  <span style={membershipInfoLabel}>Plan</span>
                  <span style={membershipInfoValue}>
                    {isLoadingUser ? "Wird geladen..." : planLabel}
                  </span>
                </div>

                <div style={membershipInfoRow}>
                  <span style={membershipInfoLabel}>Status</span>
                  <span style={membershipInfoValue}>
                    {isLoadingUser ? "Wird geladen..." : billingStatusLabel}
                  </span>
                </div>

                <div style={membershipInfoRow}>
                  <span style={membershipInfoLabel}>Intervall</span>
                  <span style={membershipInfoValue}>
                    {isLoadingUser ? "Wird geladen..." : billingIntervalLabel}
                  </span>
                </div>

                <div style={membershipInfoRow}>
                  <span style={membershipInfoLabel}>
                    {isSubscriptionCanceling ? "Läuft bis" : "Aktiv bis"}
                  </span>
                  <span style={membershipInfoValue}>
                    {isLoadingUser
                      ? "Wird geladen..."
                      : currentUser?.current_period_end
                        ? formattedCurrentPeriodEnd
                        : "Kein Enddatum verfügbar"}
                  </span>
                </div>
              </div>

              <div style={membershipActionRow}>
                <Link to="/app/billing" style={smallPrimaryButton}>
                    {isPaymentFailedCanceled
                        ? "Erneut auf Pro upgraden"
                        : isPro
                            ? "Billing ansehen"
                            : "Auf Pro upgraden"}
                </Link>

                {isSubscriptionCanceling ? (
                  <button
                    type="button"
                    onClick={handleResumeSubscription}
                    disabled={isResumingSubscription}
                    style={{
                      ...smallSecondaryButton,
                      cursor: isResumingSubscription ? "not-allowed" : "pointer",
                      opacity: isResumingSubscription ? 0.82 : 1,
                    }}
                  >
                    {isResumingSubscription
                      ? "Wird fortgeführt..."
                      : "Pro weiterführen"}
                  </button>
                ) : null}
              </div>
            </div>

            <div style={sideCard}>
              <div style={sectionEyebrow}>Sprache</div>
              <div style={sideCardTitle}>Sprache / Language</div>
              <p style={sideCardText}>
                Wähle die Sprache der Benutzeroberfläche. Deine Wahl wird in
                deinem Konto gespeichert.
              </p>

              <div style={languageToggleRow}>
                <button
                  type="button"
                  onClick={() => handleSelectLocale("de")}
                  disabled={isSavingLocale}
                  style={languageToggleButton(locale === "de", isSavingLocale)}
                >
                  Deutsch
                </button>
                <button
                  type="button"
                  onClick={() => handleSelectLocale("en")}
                  disabled={isSavingLocale}
                  style={languageToggleButton(locale === "en", isSavingLocale)}
                >
                  English
                </button>
              </div>
            </div>
          </div>
        </section>

        <section style={passwordSection} data-tour="account-profile-section">
          <div style={sectionEyebrow}>Profil</div>
          <h2 style={passwordTitle}>Dein Profil</h2>
          <p style={passwordText}>
            Benutzername, Vor- und Nachname sowie Geburtsdatum sind optional
            und können jederzeit ergänzt oder geändert werden.
          </p>

          {profileErrorMessage ? (
            <div style={passwordErrorBox}>{profileErrorMessage}</div>
          ) : null}

          {profileSuccessMessage ? (
            <div style={passwordSuccessBox}>{profileSuccessMessage}</div>
          ) : null}

          {isEditingProfile ? (
            <form onSubmit={handleSaveProfile} style={passwordForm}>
              <div style={formGrid}>
                <div style={fieldGroup}>
                  <label htmlFor="profile-username" style={fieldLabel}>
                    Benutzername
                  </label>
                  <input
                    id="profile-username"
                    type="text"
                    value={profileUsername}
                    onChange={(event) => setProfileUsername(event.target.value)}
                    style={fieldInput}
                    placeholder="z. B. max_mustermann"
                  />
                </div>

                <div style={fieldGroup}>
                  <label htmlFor="profile-first-name" style={fieldLabel}>
                    Vorname
                  </label>
                  <input
                    id="profile-first-name"
                    type="text"
                    value={profileFirstName}
                    onChange={(event) => setProfileFirstName(event.target.value)}
                    style={fieldInput}
                    placeholder="Max"
                  />
                </div>

                <div style={fieldGroup}>
                  <label htmlFor="profile-last-name" style={fieldLabel}>
                    Nachname
                  </label>
                  <input
                    id="profile-last-name"
                    type="text"
                    value={profileLastName}
                    onChange={(event) => setProfileLastName(event.target.value)}
                    style={fieldInput}
                    placeholder="Mustermann"
                  />
                </div>

                <div style={fieldGroup}>
                  <label htmlFor="profile-birth-date" style={fieldLabel}>
                    Geburtsdatum
                  </label>
                  <input
                    id="profile-birth-date"
                    type="date"
                    max={TODAY_ISO}
                    value={profileBirthDate}
                    onChange={(event) => setProfileBirthDate(event.target.value)}
                    style={fieldInput}
                  />
                </div>
              </div>

              <div style={passwordActionRow}>
                <button
                  type="submit"
                  disabled={isSavingProfile}
                  style={{
                    ...passwordPrimaryButton,
                    cursor: isSavingProfile ? "not-allowed" : "pointer",
                    opacity: isSavingProfile ? 0.82 : 1,
                    border: "none",
                  }}
                >
                  {isSavingProfile ? "Wird gespeichert..." : "Speichern"}
                </button>

                <button
                  type="button"
                  onClick={() => setIsEditingProfile(false)}
                  disabled={isSavingProfile}
                  style={{
                    ...passwordPrimaryButton,
                    background: "transparent",
                    color: theme.colors.textSecondary,
                    border: "1px solid rgba(148, 163, 184, 0.28)",
                    boxShadow: "none",
                    cursor: isSavingProfile ? "not-allowed" : "pointer",
                  }}
                >
                  Abbrechen
                </button>
              </div>
            </form>
          ) : (
            <>
              <div style={infoList}>
                <div style={infoRow}>
                  <span style={infoLabel}>Benutzername</span>
                  <span style={infoValue}>
                    {isLoadingUser
                      ? "Wird geladen..."
                      : currentUser?.username ?? "Noch nicht festgelegt"}
                  </span>
                </div>

                <div style={infoRow}>
                  <span style={infoLabel}>Vorname</span>
                  <span style={infoValue}>
                    {isLoadingUser
                      ? "Wird geladen..."
                      : currentUser?.first_name ?? "Noch nicht festgelegt"}
                  </span>
                </div>

                <div style={infoRow}>
                  <span style={infoLabel}>Nachname</span>
                  <span style={infoValue}>
                    {isLoadingUser
                      ? "Wird geladen..."
                      : currentUser?.last_name ?? "Noch nicht festgelegt"}
                  </span>
                </div>

                <div style={infoRow}>
                  <span style={infoLabel}>Geburtsdatum</span>
                  <span style={infoValue}>
                    {isLoadingUser ? "Wird geladen..." : formatBirthDateDisplay(currentUser)}
                  </span>
                </div>
              </div>

              {/* marginTop nur hier: außerhalb der Formulare fehlt der
                  passwordForm-gap, sonst kleben die Buttons an der infoList. */}
              <div style={{ ...passwordActionRow, marginTop: "24px" }}>
                <button
                  type="button"
                  onClick={handleStartEditProfile}
                  disabled={isLoadingUser}
                  style={{
                    ...passwordPrimaryButton,
                    cursor: isLoadingUser ? "not-allowed" : "pointer",
                    opacity: isLoadingUser ? 0.82 : 1,
                    border: "none",
                  }}
                >
                  Bearbeiten
                </button>

                <button
                  type="button"
                  onClick={() => navigate("/app/dashboard?startTour=1")}
                  style={{
                    ...passwordPrimaryButton,
                    background: "transparent",
                    color: theme.colors.textSecondary,
                    border: "1px solid rgba(148, 163, 184, 0.28)",
                    boxShadow: "none",
                  }}
                >
                  Tour erneut starten
                </button>
              </div>
            </>
          )}
        </section>

        {isPro ? (
          <section style={cancelSection}>
            <div style={sectionEyebrow}>Abonnement</div>
            <h2 style={cancelTitle}>Pro-Abo kündigen</h2>
            <p style={cancelText}>
              Wenn du dein Abonnement beenden möchtest, kannst du es hier
              kündigen. Dein Zugang bleibt bis zum Ende des aktuellen
              Abrechnungszeitraums bestehen.
            </p>

            {!isLoadingUser && currentUser?.current_period_end ? (
              <div style={subscriptionInfoBox}>
                {isSubscriptionCanceling
                  ? `Dein Pro-Zugang läuft noch bis ${formattedCurrentPeriodEnd}.`
                  : `Dein aktueller Abrechnungszeitraum läuft bis ${formattedCurrentPeriodEnd}.`}
              </div>
            ) : null}

            {subscriptionErrorMessage ? (
              <div style={cancelErrorBox}>{subscriptionErrorMessage}</div>
            ) : null}

            {subscriptionSuccessMessage ? (
              <div style={cancelSuccessBox}>{subscriptionSuccessMessage}</div>
            ) : null}

            <div style={cancelActionRow}>
              {isSubscriptionCanceling ? (
                <button
                  type="button"
                  onClick={handleResumeSubscription}
                  disabled={isResumingSubscription}
                  style={{
                    ...resumeButton,
                    cursor: isResumingSubscription ? "not-allowed" : "pointer",
                    opacity: isResumingSubscription ? 0.82 : 1,
                  }}
                >
                  {isResumingSubscription
                    ? "Abo wird fortgeführt..."
                    : "Kündigung zurücknehmen"}
                </button>
              ) : null}

              <button
                type="button"
                onClick={() => setIsCancelModalOpen(true)}
                disabled={isCancelingSubscription || isSubscriptionCanceling}
                style={{
                  ...cancelButton,
                  cursor:
                    isCancelingSubscription || isSubscriptionCanceling
                      ? "not-allowed"
                      : "pointer",
                  opacity:
                    isCancelingSubscription || isSubscriptionCanceling ? 0.82 : 1,
                }}
              >
                {isCancelingSubscription
                  ? "Abo wird gekündigt..."
                  : isSubscriptionCanceling
                    ? "Abo bereits vorgemerkt"
                    : "Abo kündigen"}
              </button>
            </div>
          </section>
        ) : null}

        <section style={passwordSection}>
          <div style={sectionEyebrow}>Sicherheit</div>
          <h2 style={passwordTitle}>Passwort ändern</h2>
          <p style={passwordText}>
            Wenn du dein Passwort aktualisieren möchtest, kannst du es hier direkt
            ändern. Verwende dabei ein sicheres Passwort, das du nur für dieses
            Konto nutzt.
          </p>

          {passwordErrorMessage ? (
            <div style={passwordErrorBox}>{passwordErrorMessage}</div>
          ) : null}

          {passwordSuccessMessage ? (
            <div style={passwordSuccessBox}>{passwordSuccessMessage}</div>
          ) : null}

          <form onSubmit={handleChangePassword} style={passwordForm}>
            <div style={formGrid}>
              <div style={fieldGroup}>
                <label htmlFor="current-password" style={fieldLabel}>
                  Aktuelles Passwort
                </label>
                <input
                  id="current-password"
                  type="password"
                  value={currentPassword}
                  onChange={(event) => setCurrentPassword(event.target.value)}
                  style={fieldInput}
                  placeholder="Aktuelles Passwort eingeben"
                  autoComplete="current-password"
                />
              </div>

              <div style={fieldGroup}>
                <label htmlFor="new-password" style={fieldLabel}>
                  Neues Passwort
                </label>
                <input
                  id="new-password"
                  type="password"
                  value={newPassword}
                  onChange={(event) => setNewPassword(event.target.value)}
                  style={fieldInput}
                  placeholder="Neues Passwort eingeben"
                  autoComplete="new-password"
                />
              </div>

              <div style={fieldGroup}>
                <label htmlFor="confirm-new-password" style={fieldLabel}>
                  Neues Passwort bestätigen
                </label>
                <input
                  id="confirm-new-password"
                  type="password"
                  value={confirmNewPassword}
                  onChange={(event) => setConfirmNewPassword(event.target.value)}
                  style={fieldInput}
                  placeholder="Neues Passwort wiederholen"
                  autoComplete="new-password"
                />
              </div>
            </div>

            <div style={passwordActionRow}>
              <button
                type="submit"
                disabled={isChangingPassword}
                style={{
                  ...passwordPrimaryButton,
                  cursor: isChangingPassword ? "not-allowed" : "pointer",
                  opacity: isChangingPassword ? 0.82 : 1,
                  border: "none",
                }}
              >
                {isChangingPassword
                  ? "Passwort wird geändert..."
                  : "Passwort ändern"}
              </button>
            </div>
          </form>
        </section>

        <section style={bottomSection}>
          <div style={sectionEyebrow}>Zahlungen & Rechnungen</div>
          <h2 style={bottomTitle}>
            Verwalte dein Abo, deine Rechnungen und deine Zahlungsmethode
          </h2>
          <p style={bottomText}>
            Über das sichere Kundenportal kannst du deine Zahlungsmethode
            aktualisieren, Rechnungen einsehen oder herunterladen und dein
            Abonnement eigenständig verwalten. Du wirst dafür direkt zum
            Stripe-Kundenportal weitergeleitet.
          </p>

          <div style={bottomActionRow}>
            <div style={bottomActionRow}>
        <button
            type="button"
            onClick={handleOpenCustomerPortal}
            disabled={isOpeningPortal || !canManageSubscriptionPortal}
            style={{
                ...bottomPrimaryButton,
                cursor:
                    isOpeningPortal || !canManageSubscriptionPortal
                        ? "not-allowed"
                        : "pointer",
                opacity: isOpeningPortal || !canManageSubscriptionPortal ? 0.82 : 1,
                border: "none",
              }}
            >
             {isOpeningPortal
                ? "Portal wird geöffnet..."
                : canManageSubscriptionPortal
                    ? "Abonnement verwalten"
                    : "Kein aktives Abonnement"}
        </button>
        </div>
          </div>
        </section>

        <section
          style={{
            ...bottomSection,
            border: `1px solid ${theme.colors.dangerBorder}`,
          }}
        >
          <div style={{ ...sectionEyebrow, color: theme.colors.dangerText }}>
            Danger Zone
          </div>
          <h2 style={bottomTitle}>Konto endgültig löschen</h2>
          <p style={bottomText}>
            Löscht dein Konto und alle zugehörigen Daten (Favoriten,
            Analyse-Historie, gespeicherte eigene Analysen) endgültig. Ein
            laufendes Pro-Abo wird dabei sofort beendet. Dieser Schritt kann
            nicht rückgängig gemacht werden.
          </p>

          <div style={bottomActionRow}>
            <button
              type="button"
              onClick={() => {
                setDeletePassword("");
                setDeleteErrorMessage("");
                setIsDeleteModalOpen(true);
              }}
              style={{
                padding: "13px 22px",
                borderRadius: theme.radius.md,
                border: `1px solid ${theme.colors.dangerBorder}`,
                background: theme.colors.dangerSoft,
                color: theme.colors.dangerText,
                fontWeight: 800,
                fontSize: "0.98rem",
                cursor: "pointer",
              }}
            >
              Konto löschen
            </button>
          </div>
        </section>
      </div>

      <Modal
        isOpen={isDeleteModalOpen}
        onClose={() => {
          if (!isDeletingAccount) setIsDeleteModalOpen(false);
        }}
        title="Konto wirklich löschen?"
      >
        <p
          style={{
            margin: "0 0 14px 0",
            color: theme.colors.textSecondary,
            fontSize: "0.98rem",
            lineHeight: 1.7,
          }}
        >
          Alle deine Daten werden endgültig gelöscht und ein laufendes Abo
          sofort beendet. Gib zur Bestätigung dein Passwort ein.
        </p>

        <Input
          id="deleteAccountPassword"
          type="password"
          placeholder="Dein Passwort"
          value={deletePassword}
          onChange={(event) => setDeletePassword(event.target.value)}
          disabled={isDeletingAccount}
        />

        {deleteErrorMessage ? (
          <div
            style={{
              marginTop: "12px",
              padding: "12px 14px",
              borderRadius: theme.radius.md,
              background: theme.colors.dangerSoft,
              border: `1px solid ${theme.colors.dangerBorder}`,
              color: theme.colors.dangerText,
              fontSize: "0.95rem",
              lineHeight: 1.6,
            }}
          >
            {deleteErrorMessage}
          </div>
        ) : null}

        <button
          type="button"
          onClick={handleDeleteAccount}
          disabled={isDeletingAccount}
          style={{
            marginTop: "16px",
            width: "100%",
            padding: "14px 18px",
            borderRadius: theme.radius.md,
            border: `1px solid ${theme.colors.dangerBorder}`,
            background: theme.colors.dangerSoft,
            color: theme.colors.dangerText,
            fontWeight: 800,
            fontSize: "1rem",
            cursor: isDeletingAccount ? "not-allowed" : "pointer",
            opacity: isDeletingAccount ? 0.7 : 1,
          }}
        >
          {isDeletingAccount
            ? "Konto wird gelöscht..."
            : "Endgültig löschen"}
        </button>
      </Modal>
    </>
  );
}

/** birth_date ist der neue, kanonische Wert (zeigt Alter + Datum); Konten von
 * vor der Umstellung haben oft nur das statische Legacy-`age` gesetzt - dafuer
 * unveraendert weiter dessen reine Zahl anzeigen. */
function formatBirthDateDisplay(user: CurrentUserResponse | null): string {
  if (!user) return "Noch nicht festgelegt";
  if (user.birth_date) {
    const formatted = new Date(user.birth_date).toLocaleDateString("de-DE", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
    });
    return `${formatted} (${calculateAge(user.birth_date)} Jahre)`;
  }
  if (user.age) return `${user.age} Jahre`;
  return "Noch nicht festgelegt";
}

function getPlanLabel(plan: string): string {
  switch (plan) {
    case "pro":
      return "Pro Plan";
    case "free":
      return "Free Plan";
    case "friends":
      return "Friends Plan";
    case "admin":
      return "Admin";
    default:
      return plan;
  }
}

function getBillingStatusLabel(status: string): string {
  switch (status) {
    case "active":
      return "Aktiv";
    case "canceling":
      return "Kündigt zum Periodenende";
    case "past_due":
      return "Zahlung offen";
    case "canceled":
      return "Gekündigt";
    case "payment_failed_canceled":
      return "Zahlung fehlgeschlagen (Abo beendet)";
    default:
      return status;
  }
}

function getBillingIntervalLabel(interval: string | null): string {
  if (!interval) {
    return "Kein aktives Abo";
  }

  switch (interval) {
    case "month":
      return "Monatlich";
    case "year":
      return "Jährlich";
    default:
      return interval;
  }
}

/* styles */

const heroSection = {
  position: "relative" as const,
  background: theme.colors.panel,
  borderRadius: "28px",
  padding: "34px 34px 36px",
  border: "1px solid rgba(148, 163, 184, 0.18)",
  boxShadow: "0 20px 50px rgba(0, 0, 0, 0.35)",
};

const heroBadge = {
  display: "inline-block",
  marginBottom: "16px",
  padding: "8px 12px",
  borderRadius: "999px",
  background: theme.colors.panelAlt,
  border: "1px solid rgba(96, 165, 250, 0.16)",
  color: theme.colors.chrome,
  fontSize: "0.86rem",
  fontWeight: 700,
  letterSpacing: "0.03em",
};

const heroActionRow = {
  display: "flex",
  gap: "14px",
  flexWrap: "wrap" as const,
  marginTop: "26px",
};

const primaryButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "14px 18px",
  borderRadius: "14px",
  background: theme.gradients.ctaPrimary,
  color: theme.colors.bgDeep,
  fontWeight: 800,
  fontSize: "1rem",
  boxShadow: "0 14px 30px rgba(0, 0, 0, 0.35)",
};

const secondaryButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "14px 18px",
  borderRadius: "14px",
  background: theme.colors.panel,
  color: theme.colors.textPrimary,
  fontWeight: 700,
  fontSize: "1rem",
  border: "1px solid rgba(148, 163, 184, 0.14)",
};

const contentGrid = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(min(320px, 100%), 1fr))",
  gap: "22px",
  alignItems: "stretch",
};

const mainCard = {
  position: "relative" as const,
  background: theme.colors.panel,
  borderRadius: "24px",
  padding: "28px",
  border: "1px solid rgba(148, 163, 184, 0.12)",
  boxShadow: "0 14px 34px rgba(0, 0, 0, 0.22)",
};

const sideColumn = {
  display: "flex",
  flexDirection: "column" as const,
  gap: "22px",
};

const sideCard = {
  position: "relative" as const,
  background: theme.colors.panel,
  borderRadius: "24px",
  padding: "28px",
  border: "1px solid rgba(148, 163, 184, 0.14)",
  boxShadow: "0 16px 36px rgba(0, 0, 0, 0.24)",
};

const sectionEyebrow = {
  fontSize: "0.86rem",
  fontWeight: 700,
  color: theme.colors.chrome,
  letterSpacing: "0.03em",
  textTransform: "uppercase" as const,
};

const cardTitle = {
  marginTop: "12px",
  marginBottom: "10px",
  fontSize: "1.7rem",
  fontWeight: 800,
  color: theme.colors.textPrimary,
  lineHeight: 1.3,
};

const cardText = {
  margin: "0 0 22px 0",
  color: theme.colors.textPrimary,
  fontSize: "1rem",
  lineHeight: 1.8,
};

const infoList = {
  display: "flex",
  flexDirection: "column" as const,
  gap: "14px",
};

const infoRow = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  gap: "16px",
  padding: "14px 16px",
  borderRadius: "16px",
  background: theme.colors.panelAlt,
  border: "1px solid rgba(148, 163, 184, 0.10)",
};

const infoLabel = {
  color: theme.colors.textSecondary,
  fontSize: "0.96rem",
  fontWeight: 700,
};

const infoValue = {
  color: theme.colors.textPrimary,
  fontSize: "0.96rem",
  fontWeight: 600,
};

const infoValueMuted = {
  color: theme.colors.textMuted,
  fontSize: "0.96rem",
  fontWeight: 600,
};

const statusPill = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "8px 12px",
  borderRadius: "999px",
  background: "rgba(22, 163, 74, 0.16)",
  color: theme.colors.successText,
  fontWeight: 800,
  fontSize: "0.86rem",
  border: "1px solid rgba(134, 239, 172, 0.22)",
};

const sideCardTitle = {
  marginTop: "12px",
  marginBottom: "10px",
  fontSize: "1.35rem",
  fontWeight: 800,
  color: theme.colors.textPrimary,
  lineHeight: 1.35,
};

const sideCardText = {
  margin: "0 0 18px 0",
  color: theme.colors.textPrimary,
  fontSize: "0.98rem",
  lineHeight: 1.75,
};

const planBox = {
  display: "flex",
  flexDirection: "column" as const,
  gap: "6px",
  padding: "16px 18px",
  borderRadius: "18px",
  background: theme.glass.background,
  border: "1px solid rgba(148, 163, 184, 0.14)",
  marginBottom: "18px",
};

const planName = {
  color: theme.colors.textPrimary,
  fontSize: "1.1rem",
  fontWeight: 800,
};

const planHint = {
  color: theme.colors.textSecondary,
  fontSize: "0.94rem",
  fontWeight: 600,
};

const membershipInfoList = {
  display: "flex",
  flexDirection: "column" as const,
  gap: "10px",
  marginBottom: "18px",
};

const membershipInfoRow = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  gap: "14px",
};

const membershipInfoLabel = {
  color: theme.colors.textSecondary,
  fontSize: "0.92rem",
  fontWeight: 700,
};

const membershipInfoValue = {
  color: theme.colors.textPrimary,
  fontSize: "0.92rem",
  fontWeight: 600,
};

const membershipActionRow = {
  display: "flex",
  gap: "12px",
  flexWrap: "wrap" as const,
};

const smallPrimaryButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "13px 16px",
  borderRadius: "14px",
  background: theme.gradients.ctaPrimary,
  color: theme.colors.onChrome,
  fontWeight: 800,
  fontSize: "0.96rem",
  boxShadow: "0 14px 30px rgba(0, 0, 0, 0.35)",
};

const smallSecondaryButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "13px 16px",
  borderRadius: "14px",
  background: "transparent",
  color: theme.colors.textPrimary,
  fontWeight: 800,
  fontSize: "0.96rem",
  border: "1px solid rgba(148, 163, 184, 0.16)",
};

const languageToggleRow = {
  display: "flex",
  gap: "6px",
  padding: "5px",
  borderRadius: theme.radius.pill,
  background: theme.colors.panelAlt,
  border: `1px solid ${theme.colors.border}`,
  width: "fit-content",
};

function languageToggleButton(active: boolean, disabled: boolean) {
  return {
    padding: "9px 20px",
    borderRadius: theme.radius.pill,
    border: "none",
    cursor: disabled ? "not-allowed" : "pointer",
    fontWeight: 700 as const,
    fontSize: "0.9rem",
    background: active ? theme.gradients.ctaPrimary : "transparent",
    color: active ? theme.colors.bgDeep : theme.colors.textSecondary,
    opacity: disabled ? 0.75 : 1,
  };
}

const cancelSection = {
  position: "relative" as const,
  background: theme.colors.panel,
  borderRadius: "28px",
  padding: "34px",
  border: "1px solid rgba(148, 163, 184, 0.18)",
  boxShadow: "0 20px 50px rgba(0, 0, 0, 0.35)",
};

const cancelTitle = {
  margin: "10px 0 14px 0",
  fontSize: "2rem",
  lineHeight: 1.15,
  color: theme.colors.textPrimary,
  letterSpacing: "-0.03em",
};

const cancelText = {
  margin: "0 0 24px 0",
  maxWidth: "820px",
  color: theme.colors.textPrimary,
  fontSize: "1.02rem",
  lineHeight: 1.8,
};

const subscriptionInfoBox = {
  marginBottom: "18px",
  padding: "14px 16px",
  borderRadius: "14px",
  background: theme.colors.chromeSoft,
  border: `1px solid ${theme.colors.chromeBorder}`,
  color: theme.colors.textPrimary,
  fontSize: "0.96rem",
  lineHeight: 1.7,
};

const cancelErrorBox = {
  marginBottom: "18px",
  padding: "14px 16px",
  borderRadius: "14px",
  background: theme.colors.dangerSoft,
  border: `1px solid ${theme.colors.dangerBorder}`,
  color: theme.colors.dangerText,
  fontSize: "0.96rem",
  lineHeight: 1.7,
};

const cancelSuccessBox = {
  marginBottom: "18px",
  padding: "14px 16px",
  borderRadius: "14px",
  background: theme.colors.successSoft,
  border: `1px solid ${theme.colors.successBorder}`,
  color: theme.colors.successText,
  fontSize: "0.96rem",
  lineHeight: 1.7,
};

const cancelActionRow = {
  display: "flex",
  justifyContent: "flex-start",
  gap: "14px",
  flexWrap: "wrap" as const,
};

const cancelButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "14px 18px",
  borderRadius: "14px",
  background: "linear-gradient(135deg, #b91c1c, #dc2626)",
  color: "#ffffff",
  fontWeight: 800,
  fontSize: "1rem",
  boxShadow: "0 14px 30px rgba(220, 38, 38, 0.18)",
  border: "none",
};

const resumeButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "14px 18px",
  borderRadius: "14px",
  background: theme.gradients.ctaPrimary,
  color: theme.colors.bgDeep,
  fontWeight: 800,
  fontSize: "1rem",
  boxShadow: "0 14px 30px rgba(0, 0, 0, 0.35)",
  border: "none",
};

const passwordSection = {
  position: "relative" as const,
  background: theme.colors.panel,
  borderRadius: "28px",
  padding: "34px",
  border: "1px solid rgba(148, 163, 184, 0.18)",
  boxShadow: "0 20px 50px rgba(0, 0, 0, 0.35)",
};

const passwordTitle = {
  margin: "10px 0 14px 0",
  fontSize: "2rem",
  lineHeight: 1.15,
  color: theme.colors.textPrimary,
  letterSpacing: "-0.03em",
};

const passwordText = {
  margin: "0 0 24px 0",
  maxWidth: "820px",
  color: theme.colors.textPrimary,
  fontSize: "1.02rem",
  lineHeight: 1.8,
};

const passwordErrorBox = {
  marginBottom: "18px",
  padding: "14px 16px",
  borderRadius: "14px",
  background: theme.colors.dangerSoft,
  border: `1px solid ${theme.colors.dangerBorder}`,
  color: theme.colors.dangerText,
  fontSize: "0.96rem",
  lineHeight: 1.7,
};

const passwordSuccessBox = {
  marginBottom: "18px",
  padding: "14px 16px",
  borderRadius: "14px",
  background: theme.colors.successSoft,
  border: `1px solid ${theme.colors.successBorder}`,
  color: theme.colors.successText,
  fontSize: "0.96rem",
  lineHeight: 1.7,
};

const passwordForm = {
  display: "flex",
  flexDirection: "column" as const,
  gap: "22px",
};

const formGrid = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
  gap: "18px",
};

const fieldGroup = {
  display: "flex",
  flexDirection: "column" as const,
  gap: "8px",
};

const fieldLabel = {
  color: theme.colors.textSecondary,
  fontSize: "0.94rem",
  fontWeight: 700,
};

const fieldInput = {
  width: "100%",
  padding: "14px 14px",
  borderRadius: "14px",
  border: "1px solid rgba(148, 163, 184, 0.18)",
  background: theme.colors.panelAlt,
  color: theme.colors.textPrimary,
  // >= 16px, sonst zoomt iOS Safari beim Fokussieren die Seite.
  fontSize: "max(16px, 0.98rem)",
  outline: "none",
  boxSizing: "border-box" as const,
};

const passwordActionRow = {
  display: "flex",
  justifyContent: "flex-start",
  gap: "14px",
  flexWrap: "wrap" as const,
};

const passwordPrimaryButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "14px 18px",
  borderRadius: "14px",
  background: theme.gradients.ctaPrimary,
  color: theme.colors.bgDeep,
  fontWeight: 800,
  fontSize: "1rem",
  boxShadow: "0 14px 30px rgba(0, 0, 0, 0.35)",
};

const bottomSection = {
  position: "relative" as const,
  background: theme.colors.panel,
  borderRadius: "28px",
  padding: "34px",
  border: "1px solid rgba(148, 163, 184, 0.18)",
  boxShadow: "0 20px 50px rgba(0, 0, 0, 0.35)",
  textAlign: "center" as const,
};

const bottomTitle = {
  margin: "10px 0 14px 0",
  fontSize: "2.1rem",
  lineHeight: 1.15,
  color: theme.colors.textPrimary,
  letterSpacing: "-0.03em",
};

const bottomText = {
  margin: "0 auto",
  maxWidth: "800px",
  color: theme.colors.textPrimary,
  fontSize: "1.04rem",
  lineHeight: 1.85,
};

const bottomActionRow = {
  display: "flex",
  justifyContent: "center",
  gap: "14px",
  flexWrap: "wrap" as const,
  marginTop: "26px",
};

const bottomPrimaryButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "15px 18px",
  borderRadius: "14px",
  background: theme.gradients.ctaPrimary,
  color: theme.colors.bgDeep,
  fontWeight: 800,
  fontSize: "1rem",
  boxShadow: "0 14px 30px rgba(0, 0, 0, 0.35)",
};

export default AccountPage;