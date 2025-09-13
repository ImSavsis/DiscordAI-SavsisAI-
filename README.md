### RU LANG

<div align="center">

  DiscordAI — SavsisAI

  Лёгкий Discord‑бот на Python. Только чат. Локальная ИИ‑модель через LM Studio. Поддержка Spotify без стриминга аудио в Discord.

  Документация EN • Добавить бота • Discord‑канал

</div>


### ✨ Возможности

Чат‑только: без голосовых каналов и анализа картинок.

Локальный ИИ: ответы через LM Studio (данные остаются у вас).

Русский стиль: системный промт ориентирует на краткие, доброжелательные ответы (RU).

Spotify интеграция: запуск треков на вашем аккаунте Spotify через Web API (без аудио в Discord).

/settings и !settings: настройки только для админов (ввод ключей Spotify).

Минимальные зависимости: только discord.py, aiohttp, loguru, tenacity, requests.


### 🔧 Требования

Python 3.10–3.13

Токен Discord‑бота

LM Studio (любой совместимый чат‑модель; рекомендовано google/gemma-3-4b)

(Опционально) Spotify Client ID/Secret/Refresh Token для управления музыкой

### 💬 Команды

!help — помощь

!status — статус

!ping — пинг

/settings (только админы) — настройки (ввод ключей Spotify)

!settings (только админы) — то же, в виде сообщения

Текстовая команда для Spotify: включи <название> <автор> (запускает трек на вашем Spotify‑устройстве

### 🔑 Spotify: как получить refresh_token

Создайте приложение в Spotify Dashboard. В Redirect URI укажите http://localhost:5173/callback.

https://developer.spotify.com

### 🔐 Права/безопасность

Бот отвечает на сообщения в текстовых каналах.

/settings и !settings доступны только администраторам.

Ключи Spotify хранятся локально (в config.json) или вводятся через модальное окно (ephemeral).

### 🔗 Полезные ссылки

Discord‑канал: 

https://discord.gg/eKv5VuVBKm

Добавить бота: 

https://discord.com/oauth2/authorize?client_id=1416156615601684663




### EN LANG

<div align="center">
  
### DiscordAI — SavsisAI

A lightweight Discord bot for Python. Chat only. Local AI model via LM Studio. Spotify support without streaming audio to Discord.

EN Documentation • Add Bot • Discord Channel

</div>

### ✨ Features

Chat-only: No voice channel or image analysis support.

Local AI: Responses via LM Studio (your data stays with you).

Russian Style: System prompt is oriented towards concise, friendly responses (RU).

Spotify Integration: Start tracks on your Spotify account via Web API (no audio in Discord).

/settings and !settings: Admin-only settings (for entering Spotify keys).

Minimal Dependencies: Only discord.py, aiohttp, loguru, tenacity, requests.

### 🔧 Requirements
Python 3.10–3.13


Discord Bot Token

LM Studio (any compatible chat model; google/gemma-3-4b recommended)

(Optional) Spotify Client ID/Secret/Refresh Token for music control

### 💬 Commands

!help — Help

!status — Status

!ping — Ping

/settings (admins only) — Settings (enter Spotify keys)

!settings (admins only) — Same as above, as a message

Text command for Spotify: включи <track name> <artist> (starts the track on your Spotify device)

### 🔑 Spotify: How to get a refresh_token

Create an application in the Spotify Dashboard.

Set the Redirect URI to http://localhost:5173/callback.

https://developer.spotify.com

### 🔐 Permissions/Security

The bot responds to messages in text channels.

/settings and !settings are only available to administrators.

Spotify keys are stored locally (in config.json) or entered via a modal window (ephemeral).

###🔗 Useful Links

Discord Channel:

https://discord.gg/eKv5VuVBKm

Add Bot:

https://discord.com/oauth2/authorize?client_id=1416156615601684663




