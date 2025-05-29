#!/bin/bash

echo "서버 실행중..."
uvicorn app:app --host 0.0.0.0 --port 8010 &

sleep 3

echo "ngrok 실행중..."
ngrok http --url=emoti.ngrok.app 8010
