import os, pickle, json, sqlite3, hashlib, secrets, datetime, urllib.parse, re
from functools import wraps
from flask import (Flask, render_template, request, redirect, url_for,
                   session, jsonify, g)
import sys, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from train_model import extract_features, FEATURE_NAMES, SHORTENERS

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, 'phishing.db')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

with open(MODEL_PATH, 'rb') as f:
    MODEL_DATA = pickle.load(f)
clf          = MODEL_DATA['model']
FEATURE_COLS = MODEL_DATA['feature_names']
MODEL_ACC    = MODEL_DATA['accuracy']

# ──────────────── Database ────────────────
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db: db.close()

def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL DEFAULT '',
            auth_provider TEXT NOT NULL DEFAULT 'email',
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS url_checks (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            url           TEXT NOT NULL,
            result        TEXT NOT NULL,
            confidence    REAL NOT NULL,
            features_json TEXT NOT NULL,
            reasons_json  TEXT NOT NULL,
            checked_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS contact_messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT,
            email      TEXT,
            subject    TEXT,
            message    TEXT,
            sent_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
    # Migrate: add auth_provider col if missing
    try:
        with sqlite3.connect(DB_PATH) as db:
            db.execute("ALTER TABLE users ADD COLUMN auth_provider TEXT NOT NULL DEFAULT 'email'")
    except Exception:
        pass
    print("✅ Database initialised")

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def current_user():
    uid = session.get('user_id')
    if not uid: return None
    row = get_db().execute('SELECT * FROM users WHERE id=?', (uid,)).fetchone()
    return dict(row) if row else None

def get_stats(user_id):
    checks = get_db().execute(
        'SELECT result FROM url_checks WHERE user_id=?', (user_id,)
    ).fetchall()
    total   = len(checks)
    phishing= sum(1 for c in checks if c['result']=='phishing')
    return {'total': total, 'phishing': phishing, 'legit': total-phishing, 'model_acc': f"{MODEL_ACC*100:.1f}%"}

# ──────────────── Reasons ─────────────────
def build_reasons(features, result):
    f = features
    phishing_flags, legit_flags = [], []
    if f['has_ip']:             phishing_flags.append("🔴 IP address used instead of a domain name — classic phishing trick to hide identity.")
    if f['is_shortener']:       phishing_flags.append("🔴 URL shortener detected — often used to mask the real malicious destination.")
    if f['suspicious_tld']:     phishing_flags.append("🔴 Suspicious TLD (.tk .ml .ga .xyz etc.) commonly abused by phishing sites.")
    if f['has_suspicious_words']:phishing_flags.append("🔴 Social-engineering keywords found ('login', 'verify', 'secure', 'account') — classic bait.")
    if f['brand_in_subdomain']: phishing_flags.append("🔴 Brand name in subdomain rather than root domain — impersonation attempt.")
    if f['url_length']>100:     phishing_flags.append(f"🟠 Unusually long URL ({f['url_length']} chars) — often obfuscated to hide malicious destination.")
    if f['num_hyphens']>3:      phishing_flags.append(f"🟠 Many hyphens ({f['num_hyphens']}) — frequently used to mimic legitimate brands.")
    if f['num_subdomains']>2:   phishing_flags.append(f"🟠 Deep subdomain nesting ({f['num_subdomains']} levels) — used to make fake domains look real.")
    if f['num_at']>0:           phishing_flags.append("🔴 '@' symbol in URL — browsers ignore everything before '@', redirecting to a malicious host.")
    if f['double_slash_in_path']:phishing_flags.append("🟠 Double slash '//' in path — used to confuse URL parsers.")
    if f['has_hex']:            phishing_flags.append("🟠 Hex/percent encoding — often used to obfuscate malicious URLs.")
    if not f['uses_https']:     phishing_flags.append("🔴 No HTTPS — connection is unencrypted; credentials sent in plain text.")
    if f['long_domain']:        phishing_flags.append(f"🟠 Very long domain ({f['domain_length']} chars) — real domains are usually short and memorable.")
    if f['digit_ratio']>0.2:    phishing_flags.append(f"🟠 High digit ratio ({f['digit_ratio']:.0%}) — random number strings are a common phishing pattern.")
    if f['has_port']:           phishing_flags.append("🟠 Non-standard port — legitimate sites rarely expose port numbers in URLs.")
    if f['uses_https']:         legit_flags.append("✅ HTTPS encryption present — site uses a security certificate.")
    if f['num_subdomains']<=1:  legit_flags.append("✅ Clean subdomain structure — no suspicious nesting.")
    if not f['has_ip']:         legit_flags.append("✅ Proper domain name — no raw IP address.")
    if not f['suspicious_tld']: legit_flags.append("✅ Standard top-level domain — associated with trustworthy organisations.")
    if not f['has_suspicious_words']: legit_flags.append("✅ No social-engineering keywords.")
    if not f['is_shortener']:   legit_flags.append("✅ Not a URL shortener — destination is transparent.")
    if f['url_length']<=75:     legit_flags.append("✅ URL length is normal and not obfuscated.")
    if not f['brand_in_subdomain']: legit_flags.append("✅ No brand impersonation in subdomains.")
    return {
        'phishing_flags': phishing_flags, 'legit_flags': legit_flags,
        'summary': f"{'⚠️ Phishing' if result=='phishing' else '✅ Legitimate'} URL — {len(phishing_flags)} warning(s), {len(legit_flags)} trust indicator(s)."
    }

# ══════════════════════════════════════════
#   PUBLIC PAGES
# ══════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET','POST'])
def contact():
    sent = False; error = None; form_data = None
    if request.method == 'POST':
        name    = request.form.get('name','').strip()
        email   = request.form.get('email','').strip()
        subject = request.form.get('subject','general')
        message = request.form.get('message','').strip()
        if not name or not email or not message:
            error = 'Please fill in all required fields.'
            form_data = {'name':name,'email':email,'message':message}
        else:
            get_db().execute(
                'INSERT INTO contact_messages (name,email,subject,message) VALUES (?,?,?,?)',
                (name, email, subject, message)
            )
            get_db().commit()
            sent = True
    return render_template('contact.html', sent=sent, error=error, form=form_data)

@app.route('/help')
def help_page():
    return render_template('help.html')

# ══════════════════════════════════════════
#   AUTH
# ══════════════════════════════════════════

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        email    = request.form.get('email','').strip()
        password = request.form.get('password','')
        confirm  = request.form.get('confirm','')
        errors = []
        if len(username)<3:    errors.append('Username must be ≥ 3 characters.')
        if '@' not in email:   errors.append('Invalid email address.')
        if len(password)<6:    errors.append('Password must be ≥ 6 characters.')
        if password!=confirm:  errors.append('Passwords do not match.')
        if not errors:
            try:
                get_db().execute(
                    "INSERT INTO users (username,email,password_hash,auth_provider) VALUES (?,?,?,'email')",
                    (username, email, hash_pw(password))
                )
                get_db().commit()
                return redirect(url_for('login', success='Account created! Please sign in.'))
            except sqlite3.IntegrityError:
                errors.append('Username or email already exists.')
        return render_template('signup.html', errors=errors, username=username, email=email)
    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    success = request.args.get('success')
    if request.method == 'POST':
        email    = request.form.get('email','').strip()
        password = request.form.get('password','')
        user = get_db().execute(
            'SELECT * FROM users WHERE email=? AND password_hash=?',
            (email, hash_pw(password))
        ).fetchone()
        if user:
            session['user_id']  = user['id']
            session['username'] = user['username']
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='Invalid email or password.')
    return render_template('login.html', success=success)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ── Google OAuth (mock — shows info page since no real client ID needed) ──
@app.route('/auth/google')
def google_login():
    """
    In a real app this would redirect to Google's OAuth endpoint.
    Here we show a friendly page explaining what would happen,
    and simulate a Google login by creating/fetching a demo Google account.
    """
    # Create a demo "Google user" for demonstration
    demo_email = "demo.google@gmail.com"
    demo_name  = "GoogleUser"
    existing = get_db().execute('SELECT * FROM users WHERE email=?', (demo_email,)).fetchone()
    if existing:
        session['user_id']  = existing['id']
        session['username'] = existing['username']
    else:
        try:
            get_db().execute(
                "INSERT INTO users (username,email,password_hash,auth_provider) VALUES (?,?,'','google')",
                (demo_name, demo_email)
            )
            get_db().commit()
            user = get_db().execute('SELECT * FROM users WHERE email=?', (demo_email,)).fetchone()
            session['user_id']  = user['id']
            session['username'] = user['username']
        except sqlite3.IntegrityError:
            user = get_db().execute('SELECT * FROM users WHERE email=?', (demo_email,)).fetchone()
            session['user_id']  = user['id']
            session['username'] = user['username']
    return redirect(url_for('dashboard'))

# ══════════════════════════════════════════
#   ACCOUNT
# ══════════════════════════════════════════

@app.route('/account')
@login_required
def account():
    user   = current_user()
    stats  = get_stats(user['id'])
    recent = get_db().execute(
        'SELECT * FROM url_checks WHERE user_id=? ORDER BY checked_at DESC LIMIT 8',
        (user['id'],)
    ).fetchall()
    recent_checks = [dict(r) for r in recent]
    # Pass flash-style query params
    profile_saved = request.args.get('profile_saved')
    profile_error = request.args.get('profile_error')
    pw_saved      = request.args.get('pw_saved')
    pw_error      = request.args.get('pw_error')
    return render_template('account.html', user=user, stats=stats, recent_checks=recent_checks,
                           profile_saved=profile_saved, profile_error=profile_error,
                           pw_saved=pw_saved, pw_error=pw_error)

@app.route('/account/update', methods=['POST'])
@login_required
def account_update():
    user     = current_user()
    username = request.form.get('username','').strip()
    email    = request.form.get('email','').strip()
    if len(username)<3:
        return redirect(url_for('account', profile_error='Username must be ≥ 3 chars.', tab='profile'))
    try:
        get_db().execute('UPDATE users SET username=?,email=? WHERE id=?', (username, email, user['id']))
        get_db().commit()
        session['username'] = username
        return redirect(url_for('account', profile_saved='1', tab='profile'))
    except sqlite3.IntegrityError:
        return redirect(url_for('account', profile_error='Username or email already taken.', tab='profile'))

@app.route('/account/password', methods=['POST'])
@login_required
def account_password():
    user    = current_user()
    current = request.form.get('current','')
    new_pw  = request.form.get('new_pw','')
    confirm = request.form.get('confirm','')
    if user['password_hash'] != hash_pw(current):
        return redirect(url_for('account', pw_error='Current password is incorrect.', tab='security'))
    if len(new_pw)<6:
        return redirect(url_for('account', pw_error='New password must be ≥ 6 characters.', tab='security'))
    if new_pw != confirm:
        return redirect(url_for('account', pw_error='Passwords do not match.', tab='security'))
    get_db().execute('UPDATE users SET password_hash=? WHERE id=?', (hash_pw(new_pw), user['id']))
    get_db().commit()
    return redirect(url_for('account', pw_saved='1', tab='security'))

# ══════════════════════════════════════════
#   DASHBOARD
# ══════════════════════════════════════════

@app.route('/dashboard')
@login_required
def dashboard():
    user  = current_user()
    rows  = get_db().execute(
        'SELECT * FROM url_checks WHERE user_id=? ORDER BY checked_at DESC LIMIT 50',
        (user['id'],)
    ).fetchall()
    checks = []
    for r in rows:
        d = dict(r)
        d['reasons']  = json.loads(d['reasons_json'])
        d['features'] = json.loads(d['features_json'])
        checks.append(d)
    stats = {'total': len(checks),
             'phishing': sum(1 for c in checks if c['result']=='phishing'),
             'legit':    sum(1 for c in checks if c['result']=='legitimate'),
             'model_acc': f"{MODEL_ACC*100:.1f}%"}
    return render_template('dashboard.html', user=user, checks=checks, stats=stats)

@app.route('/check')
@login_required
def check_page():
    return render_template('check.html', user=current_user())

# ══════════════════════════════════════════
#   API
# ══════════════════════════════════════════

@app.route('/api/me')
@login_required
def api_me():
    user = current_user()
    return jsonify({'username': user['username'], 'email': user['email']})

@app.route('/api/analyse', methods=['POST'])
@login_required
def analyse():
    data = request.get_json()
    url  = (data or {}).get('url','').strip()
    if not url: return jsonify({'error': 'No URL provided'}), 400
    if not url.startswith(('http://','https://')): url = 'http://' + url

    feats  = extract_features(url)
    X      = pd.DataFrame([feats])[FEATURE_COLS]
    pred   = clf.predict(X)[0]
    proba  = clf.predict_proba(X)[0]
    result = 'phishing' if pred==1 else 'legitimate'
    conf   = float(proba[pred])
    reasons= build_reasons(feats, result)
    user   = current_user()

    get_db().execute(
        'INSERT INTO url_checks (user_id,url,result,confidence,features_json,reasons_json) VALUES (?,?,?,?,?,?)',
        (user['id'], url, result, conf, json.dumps(feats), json.dumps(reasons))
    )
    get_db().commit()

    highlights = [
        {'name':'URL Length',         'value':feats['url_length'],                                         'flag':feats['url_length']>100},
        {'name':'Uses HTTPS',         'value':'Yes' if feats['uses_https']       else 'No',                'flag':not feats['uses_https']},
        {'name':'IP in Domain',       'value':'Yes' if feats['has_ip']           else 'No',                'flag':feats['has_ip']},
        {'name':'URL Shortener',      'value':'Yes' if feats['is_shortener']     else 'No',                'flag':feats['is_shortener']},
        {'name':'Suspicious TLD',     'value':'Yes' if feats['suspicious_tld']   else 'No',                'flag':feats['suspicious_tld']},
        {'name':'Sus. Keywords',      'value':'Yes' if feats['has_suspicious_words'] else 'No',            'flag':feats['has_suspicious_words']},
        {'name':'Subdomains',         'value':feats['num_subdomains'],                                     'flag':feats['num_subdomains']>2},
        {'name':'Hyphens',            'value':feats['num_hyphens'],                                        'flag':feats['num_hyphens']>3},
        {'name':'Special Chars',      'value':feats['num_special_chars'],                                  'flag':feats['num_special_chars']>5},
        {'name':'Brand Impersonation','value':'Yes' if feats['brand_in_subdomain'] else 'No',              'flag':feats['brand_in_subdomain']},
    ]
    return jsonify({'url':url,'result':result,'confidence':round(conf*100,1),
                    'reasons':reasons,'features':highlights,'model_accuracy':f"{MODEL_ACC*100:.1f}%"})

@app.route('/api/delete/<int:check_id>', methods=['DELETE'])
@login_required
def delete_check(check_id):
    user = current_user()
    get_db().execute('DELETE FROM url_checks WHERE id=? AND user_id=?', (check_id, user['id']))
    get_db().commit()
    return jsonify({'ok': True})

@app.route('/api/clear-history', methods=['POST'])
@login_required
def clear_history():
    user = current_user()
    get_db().execute('DELETE FROM url_checks WHERE user_id=?', (user['id'],))
    get_db().commit()
    return jsonify({'ok': True})

@app.route('/api/delete-account', methods=['POST'])
@login_required
def delete_account():
    user = current_user()
    get_db().execute('DELETE FROM url_checks WHERE user_id=?', (user['id'],))
    get_db().execute('DELETE FROM users WHERE id=?', (user['id'],))
    get_db().commit()
    session.clear()
    return jsonify({'ok': True})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
    
    