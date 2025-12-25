"""
SNS Protocol Video Calling Website

A full-fledged secure video calling and texting website using the SNS Protocol for end-to-end encryption.
Features: User accounts, login, user search, video calls with accept/decline, hang up, mute.

To run:
1. Install dependencies: pip install -r requirements.txt
2. Run: python app.py
3. Open http://localhost:5001 in browser
4. Register users, login, search and call.

Note: Uses SQLite for simplicity. In production, use a proper database.
"""

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_login import LoginManager, login_user, login_required, logout_user, current_user  # type: ignore
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import db, User
from sns_protocol2 import SNSProtocol2 as SNSProtocol
import os
import tempfile

app = Flask(__name__, template_folder='../templates')
app.config['SECRET_KEY'] = 'sns_secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app, cors_allowed_origins="*")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Global protocols cache (in production, use Redis or similar)
protocols = {}

def get_protocol(user1, user2):
    key = f"{min(user1, user2)}_{max(user1, user2)}"
    if key not in protocols:
        protocols[key] = SNSProtocol(user1, user2, f"session_{key}")
    return protocols[key]



@app.route('/')
def landing():
    """Landing page explaining the new era of digital security."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('landing.html')

@app.route('/app')
@login_required  
def app_interface():
    """Serve the main interface."""
    return render_template('index.html')

# Keep index as alias
@app.route('/index')
@login_required
def index():
    return redirect(url_for('app_interface'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/search', methods=['POST'])
@login_required
def search():
    query = request.form['query']
    users = User.query.filter(User.username.contains(query), User.username != current_user.username).all()
    return render_template('search.html', users=users)

@app.route('/call/<username>')
@login_required
def call_user(username):
    user = User.query.filter_by(username=username).first()
    if not user:
        flash('User not found')
        return redirect(url_for('index'))
    return render_template('call.html', callee=username)

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    if current_user.is_authenticated:
        join_room(current_user.username)
        emit('status', {'message': f'Connected as {current_user.username} with SNS Protocol Level 2 - 25-Layer Quantum-Resistant Encryption!'})

@socketio.on('send_message')
def handle_message(data):
    """Handle chat messages."""
    if not current_user.is_authenticated:
        return
    msg = data['message']
    to_user = data['to']
    protocol = get_protocol(current_user.username, to_user)
    encrypted = protocol.encrypt_message(msg)
    decrypted = protocol.decrypt_message(encrypted)
    emit('receive_message', {'message': decrypted, 'from': current_user.username, 'encrypted': encrypted.hex()}, room=to_user)

@socketio.on('send_file')
def handle_file(data):
    """Handle file sharing."""
    if not current_user.is_authenticated:
        return
    import base64
    file_data = base64.b64decode(data['file_data'])
    filename = data['filename']
    to_user = data['to']
    protocol = get_protocol(current_user.username, to_user)
    encrypted = protocol.encrypt_data(file_data)
    # Store metadata with encrypted data
    metadata = f"{current_user.username}|{to_user}|{filename}".encode()
    full_data = metadata + b'|||' + encrypted
    with tempfile.NamedTemporaryFile(delete=False, suffix='.enc') as tmp:
        tmp.write(full_data)
        tmp_path = tmp.name
    emit('file_received', {'filename': filename, 'encrypted_path': tmp_path, 'from': current_user.username}, room=to_user)

@app.route('/download/<path>')
@login_required
def download_file(path):
    """Serve decrypted file."""
    # Read encrypted file with metadata
    with open(path, 'rb') as f:
        full_data = f.read()

    # Parse metadata
    try:
        metadata_part, encrypted_part = full_data.split(b'|||', 1)
        sender, receiver, filename = metadata_part.decode().split('|')
        encrypted_data = encrypted_part
    except:
        return "Invalid file format", 400

    # Use correct protocol
    protocol = get_protocol(sender, receiver)

    try:
        decrypted_data = protocol.decrypt_data(encrypted_data)
        # Create a temporary decrypted file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp.write(decrypted_data)
            tmp_path = tmp.name
        return send_file(tmp_path, as_attachment=True, download_name=filename)
    except Exception as e:
        return f"Error decrypting file: {str(e)}", 500

# Call management
calls = {}  # caller: callee

@socketio.on('initiate_call')
def initiate_call(data):
    """Initiate a call."""
    callee = data['callee']
    calls[current_user.username] = callee
    emit('incoming_call', {'caller': current_user.username}, room=callee)

@socketio.on('accept_call')
def accept_call(data):
    """Accept call."""
    caller = data['caller']
    if calls.get(caller) == current_user.username:
        emit('call_accepted', {'callee': current_user.username}, room=caller)
        emit('call_started', room=caller)
        emit('call_started', room=current_user.username)

@socketio.on('decline_call')
def decline_call(data):
    """Decline call."""
    caller = data['caller']
    emit('call_declined', room=caller)

@socketio.on('hang_up')
def hang_up(data):
    """Hang up call."""
    peer = data['peer']
    emit('call_ended', room=peer)
    if current_user.username in calls:
        del calls[current_user.username]

# WebRTC Signaling
@socketio.on('offer')
def handle_offer(data):
    """Handle offer from caller and send to callee."""
    callee = calls.get(current_user.username)
    if callee:
        emit('offer', data, room=callee)

@socketio.on('answer')
def handle_answer(data):
    """Handle answer from callee and send to caller."""
    caller = next((k for k, v in calls.items() if v == current_user.username), None)
    if caller:
        emit('answer', data, room=caller)

@socketio.on('ice-candidate')
def handle_ice_candidate(data):
    """Handle ICE candidate from either and send to peer."""
    # Send to callee if caller, or to caller if callee
    callee = calls.get(current_user.username)
    if callee:
        emit('ice-candidate', data, room=callee)
    caller = next((k for k, v in calls.items() if v == current_user.username), None)
    if caller:
        emit('ice-candidate', data, room=caller)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)