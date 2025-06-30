## Fall Detection System

A real-time fall detection application with a graphical user interface built using CustomTkinter, powered by YOLO object detection and integrated with Discord notifications. It captures video, identifies human falls, and sends alerts (images or video clips) to a specified Discord channel.

---

### 📦 Features

* **Real-Time detection** using YOLOv8 for fast and accurate person detection.
* **Bed area configuration**: Manually draw bed regions or use AI-assisted selection via Groq API.
* **Fall logic**: Detects when a person transitions from upright to horizontal posture off the bed beyond a configurable threshold.
* **Infrared mode**: Apply a COLORMAP\_INFERNO effect for low-light scenarios.
* **Discord Alerts**: Automatically send image snapshots and short video clips upon fall detection.
* **History Management**: Automatically keeps only the latest 5 messages in the Discord channel.
* **Full-Screen GUI**: Clean, dark-themed interface with clock, status, and intuitive navigation.

---

### 🛠️ Prerequisites

* **Python 3.8+**
* A camera device compatible with OpenCV
* **Discord Bot** with a token and channel permissions
* Groq API key for AI-assisted bed detection

---

### ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/fall-detection-system.git
   cd fall-detection-system
   ```

2. **Create and activate** a virtual environment

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   * Duplicate `env.example` to `.env`
   * Fill in your credentials:

     ```env
     DISCORD_TOKEN=your_discord_bot_token
     CHANNEL_ID=your_discord_channel_id
     GROQ_API_KEY=your_groq_api_key
     ```

5. **Run the application**

   ```bash
   python fall_detection_app.py
   ```

---




### 🚀 Usage

1. **Launch** the application:

   ```bash
   python fall_detection_app.py
   ```

2. **Navigate**:

   * **Home**: Overview and start detection
   * **Settings**: Configure camera index, fall threshold, toggle Discord notifications, and define bed areas manually or via AI
   * **Detection**: View real-time feed, toggle infrared mode, and monitor status

3. **Alerts**:

   * Upon a detected fall, an image snapshot and a short video clip (3 seconds buffered) are sent to your specified Discord channel.

---



