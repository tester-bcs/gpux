#!/bin/bash
# Add WanGP Studio location to nginx on vps-ru (idempotent-ish)
set -e
CONF=/etc/nginx/sites-enabled/wangp

# 1. htpasswd (only if missing)
if [ ! -f /etc/nginx/.htpasswd_studio ]; then
  HASH=$(openssl passwd -apr1 'studio2026')
  echo "avk:$HASH" > /etc/nginx/.htpasswd_studio
  chmod 640 /etc/nginx/.htpasswd_studio
  chgrp www-data /etc/nginx/.htpasswd_studio
  echo "htpasswd created (user: avk)"
fi

# 2. location block (only if not already present)
if ! grep -q "location /studio/" $CONF; then
  # insert before the existing "location / {" block
  python3 - <<'EOF'
conf = '/etc/nginx/sites-enabled/wangp'
block = '''    location /studio/ {
        auth_basic "WanGP Studio";
        auth_basic_user_file /etc/nginx/.htpasswd_studio;
        proxy_pass http://100.64.0.1:8095/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

'''
with open(conf) as f:
    content = f.read()
content = content.replace('    location / {', block + '    location / {', 1)
with open(conf, 'w') as f:
    f.write(content)
print('location /studio/ added')
EOF
else
  echo "location already present"
fi

# 3. test and reload
nginx -t && systemctl reload nginx && echo "nginx reloaded OK"
curl -s -o /dev/null -w "local check /studio/: %{http_code}\n" -u avk:studio2026 http://127.0.0.1/studio/api/status