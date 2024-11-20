import os
import sys
import smtplib
from email import encoders
from dotenv import load_dotenv
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from . import assets_dir
from .logger import create_logger

logger = create_logger(l_name="zt_mailer")

load_dotenv()

default_target_email = os.getenv("TARGET_EMAIL")


def send_mail(target_email_addr: str = default_target_email) -> None:
    """
    Sends an email

    Args:
        target_email_address: Email ID of target user

    Returns:
        None
    """
    MAILGUN_EMAIL = os.getenv("MAILGUN_EMAIL")
    MAILGUN_PWD = os.getenv("MAILGUN_PWD")

    try:
        server = smtplib.SMTP("smtp.mailgun.org", 587)
        server.login(MAILGUN_EMAIL, MAILGUN_PWD)
    except:
        logger.error("Error Connecting To Mail Server")

    subject = "Experiment Results:"
    body = f"Yo\n\nThis is a test email\n\nThank you!\n\nCheers,\n\nZeeTee"
    msg = f"Subject: {subject}\n\n{body}"

    try:
        server.sendmail(MAILGUN_EMAIL, target_email_addr, msg)
    except:
        logger.error("Error Sending Mail")

    server.quit()


def send_mail_with_attachment(folder_name: str = assets_dir, target_email_addr: str = default_target_email) -> None:
    """
    Sends an email with all the files present in a particular folder

    Args:
        folder_name: Name of folder containing the files
        target_email_addr: Email ID of target user

    Returns:
        None
    """
    MAILGUN_EMAIL = os.getenv("MAILGUN_EMAIL")
    MAILGUN_PWD = os.getenv("MAILGUN_PWD")

    if not MAILGUN_EMAIL or not MAILGUN_PWD:
        logger.error("Mailgun Credentials Are Not Set In Environment Variables")
        return

    # Creates the mail object
    msg = MIMEMultipart()
    msg["From"] = MAILGUN_EMAIL
    msg["To"] = target_email_addr
    msg["Subject"] = "Experiment Results:"

    body = "Yo\n\nPlease find attached the results you requested from your experiment\n\nThank you!\n\nCheers,\n\nZeeTee"
    msg.attach(MIMEText(body, "plain"))

    # Get the directory of the current file
    src_dir = os.path.dirname(__file__)
    # Get path to assets directory
    folder_path = os.path.abspath(os.path.join(src_dir, folder_name))

    if os.path.exists(folder_path):
        logger.debug("Path Found! Attaching Files...")

    else:
        logger.error("Path Not Found! Please Give A Valid Folder Name Within The Current [SRC] Directory...")
        return

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            f_check = file_name.split("_")[0]
            if f_check == "preprocessed":
                logger.warning("Ignoring Preprocessed Files: Size Too Big!")
                continue

            file_path = os.path.join(root, file_name)

            try:
                # Open the file in binary mode
                with open(file_path, "rb") as attachment:
                    # Create a MIMEBase object for the attachment
                    att = MIMEBase("application", "octet-stream")
                    att.set_payload(attachment.read())

                # Encode the attachment in base64
                encoders.encode_base64(att)
                # Add the header with the filename
                att.add_header("Content-Disposition", f"attachment; filename= {file_name}")
                # Attach the file to the email message
                msg.attach(att)

            except Exception as e:
                print(f"Could not attach file {file_name}: {e}")

    # Connect to the SMTP server and send the email
    try:
        server = smtplib.SMTP("smtp.mailgun.org", 587)
        server.starttls()
        server.login(MAILGUN_EMAIL, MAILGUN_PWD)

        text = msg.as_string()
        server.sendmail(MAILGUN_EMAIL, target_email_addr, text)

        logger.info("Email sent successfully.")

    except Exception as e:
        logger.error(f"Error sending mail: {e}")

    finally:
        # Ensure the SMTP server is properly closed
        try:
            server.quit()

        except Exception as e:
            logger.error(f"Error closing mail server: {e}")


if __name__ == "__main__":
    logger.info("********************MAILER[LOCAL_TESTING]*********************")

    logger.info("Sending Basic Email (NO ATTACHMENTS)...")
    send_mail()

    logger.info("Sending Email With Attachment...")
    send_mail_with_attachment()