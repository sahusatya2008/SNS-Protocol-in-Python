import base64
from flask_sqlalchemy import SQLAlchemy  # type: ignore
from flask_login import UserMixin  # type: ignore
from werkzeug.security import generate_password_hash, check_password_hash
from sns_protocol import SNSProtocol

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=False)

    def set_password(self, password):
        # Encrypt password with SNS (using a global key for demo)
        protocol = SNSProtocol("admin", "user", "password_seed")
        encrypted = protocol.encrypt_message(password)
        self.password_hash = base64.b64encode(encrypted).decode('utf-8')

    def check_password(self, password):
        try:
            protocol = SNSProtocol("admin", "user", "password_seed")
            encrypted = base64.b64decode(self.password_hash.encode('utf-8'))
            decrypted = protocol.decrypt_message(encrypted)
            return decrypted == password
        except:
            return False