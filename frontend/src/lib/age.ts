export const MIN_AGE = 16;

/** Ganze Jahre seit birthDate (ISO "YYYY-MM-DD"), korrekt für noch nicht
 * erreichte Geburtstage im laufenden Jahr — mirrort
 * api/schemas/user.py::calculate_age. */
export function calculateAge(birthDate: string): number {
  const birth = new Date(birthDate);
  const today = new Date();
  let years = today.getFullYear() - birth.getFullYear();
  const hasHadBirthdayThisYear =
    today.getMonth() > birth.getMonth() ||
    (today.getMonth() === birth.getMonth() && today.getDate() >= birth.getDate());
  if (!hasHadBirthdayThisYear) years -= 1;
  return years;
}
