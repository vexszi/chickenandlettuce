import os
import base64
from email.message import EmailMessage

from fastmcp import FastMCP
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

mcp = FastMCP("Hospital Server")


def get_gmail_service():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file(
            "token.json",
            SCOPES
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        else:
            # IMPORTANT: don't launch an interactive browser OAuth flow
            # from inside a live request (confirm_booking() in book.py
            # calls this synchronously while a patient is waiting on a
            # response -- InstalledAppFlow.run_local_server() would spin
            # up a local HTTP server and block that request thread
            # indefinitely waiting on a browser that no one is looking
            # at). Do the one-time interactive auth ahead of time instead:
            #   python mcp_server/test_mcp.py --send
            # which will open the consent screen once and write token.json.
            raise RuntimeError(
                "No valid Gmail token.json found. Run "
                "`python mcp_server/test_mcp.py --send` once from the repo "
                "root (with credentials.json present) to complete the "
                "interactive OAuth consent and cache token.json. "
                "Do not attempt interactive auth from a live server request."
            )

    return build("gmail", "v1", credentials=creds)


@mcp.tool()
def send_confirmation_email(
    patient_email: str,
    patient_name: str,
    doctor_name: str,
    appointment_date: str,
    appointment_time: str,
) -> str:
    """Send a hospital appointment confirmation email."""

    message = EmailMessage()

    message["To"] = patient_email
    message["Subject"] = "Hospital Appointment Confirmation"

    message.set_content(
        f"""Hello {patient_name},

Your appointment has been confirmed.

Doctor: {doctor_name}
Date: {appointment_date}
Time: {appointment_time}

Thank you,
Hospital Appointment System
"""
    )

    encoded_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode()

    service = get_gmail_service()

    service.users().messages().send(
        userId="me",
        body={"raw": encoded_message}
    ).execute()

    return f"Confirmation email sent successfully to {patient_email}"


if __name__ == "__main__":
    mcp.run()