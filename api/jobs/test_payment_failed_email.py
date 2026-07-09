from api.services.email_service import send_payment_failed_email


if __name__ == "__main__":
    success = send_payment_failed_email("gecen.efe1308@gmail.com")
    print(f"email_sent={success}")