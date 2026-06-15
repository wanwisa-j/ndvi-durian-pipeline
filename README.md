# NDVI Pipeline — วิธี Setup บน Server ใหม่

## สิ่งที่ต้องมีก่อน

- Docker Engine ≥ 24
- Docker Compose plugin (`docker compose` ไม่ใช่ `docker-compose`)
- RAM ≥ 30 GB (แนะนำ ≥ 64 GB สำหรับ N_JOBS=8)
- พื้นที่ disk ≥ 20 GB สำหรับ image

---

## 1. Clone / copy โค้ดลง server

```bash
# ถ้าใช้ git
git clone <repo-url> ndvi-pipeline
cd ndvi-pipeline

# หรือ scp จากเครื่องตัวเอง
scp -r ./ndvi-pipeline user@server:/home/user/ndvi-pipeline
ssh user@server
cd /home/user/ndvi-pipeline
```

ตรวจสอบว่าไฟล์ครบ:

```
ndvi-pipeline/
├── Dockerfile
├── docker-compose.yml
├── ndvi_pipeline.py
├── requirements.txt
├── run.sh
└── envs/
    ├── chanthaburi.env
    └── rayong.env        ← สร้างตาม template ด้านล่าง
```

---

## 2. สร้าง env file สำหรับแต่ละจังหวัด

สร้างโฟลเดอร์ `envs/` แล้วสร้างไฟล์ per-province:

```bash
mkdir -p envs
```

**ตัวอย่าง `envs/rayong.env`:**

```env
# --- Required ---
PROVINCE=rayong
GOLDEN_DURIAN_PATH=/fs2/wanwisa/durian/input/rayong.parquet
OUTPUT_DIR=/fs2/wanwisa/durian/output/ndvi

# --- Optional (ถ้าไม่ใส่จะใช้ default) ---
SENTINEL2_TILE_PATH=/fs2/angkanap/00_MAP_TH/03_SENTINEL-2_TILES/sentinel_2_index_shapefile.shp
S2_FOLDER=/fs2/sentinel2/tiles
TREE=segment
YEAR_START=2019
YEAR_END=2026
BAND1=08
BAND2=04
N_JOBS=8
MAX_RAM_GB=100
```

> **หมายเหตุ:** คัดลอก template นี้แล้วเปลี่ยนแค่ `PROVINCE`, `GOLDEN_DURIAN_PATH`, `OUTPUT_DIR`

---

## 3. Build Docker image

ครั้งแรกจะนานประมาณ **15–30 นาที** เพราะต้อง compile GDAL จาก source

```bash
docker compose build
```

ตรวจสอบว่า build สำเร็จ:

```bash
docker images | grep ndvi-pipeline
# ควรเห็น ndvi-pipeline:latest
```

---

## 4. รัน Pipeline

### ใช้ docker compose

```bash
docker compose --env-file envs/rayong.env up -d
docker compose --env-file envs/rayong.env logs -f
```

---

## 5. Debug / ตรวจสอบปัญหา

### เปิด shell เข้าไปใน container

```bash
./run.sh shell rayong
```

จากนั้นทดสอบ GDAL:

```bash
python3 -c "from osgeo import gdal; print('GDAL OK:', gdal.VersionInfo())"
```

ทดสอบ env vars:

```bash
env | grep -E "PROVINCE|GOLDEN|OUTPUT|S2_FOLDER"
```

### ดู log ย้อนหลัง

Log จะอยู่ที่ `$OUTPUT_DIR/<province>/logs/ndvi_<province>_<timestamp>.log`

```bash
# ดู log ล่าสุด
./run.sh tail rayong

# หรือดูตรงๆ
ls /fs2/wanwisa/durian/output/ndvi/rayong/logs/
tail -f /fs2/wanwisa/durian/output/ndvi/rayong/logs/ndvi_rayong_*.log
```

### ดู memory ขณะรัน

```bash
docker stats ndvi_rayong
```

---

## 6. โครงสร้าง Output

```
$OUTPUT_DIR/
└── <province>/
    ├── logs/
    │   └── ndvi_<province>_<YYYYMMDD_HHMMSS>.log
    └── <province>_segment_ndvi_<tile>_<polygon_idx>.parquet
```

แต่ละ parquet จะมี columns:
- `x`, `y` —좌표 EPSG ของ tile นั้น
- `geometry` — Point geometry
- `YYYY/MM/DD` — NDVI value (float32) ของแต่ละวันที่มีภาพ

---

## Troubleshooting

| อาการ | สาเหตุที่เป็นไปได้ | วิธีแก้ |
|---|---|---|
| `EnvironmentError: 'PROVINCE' is required` | ไม่ได้ส่ง env file | ใช้ `--env-file envs/<province>.env` หรือ `./run.sh run <province>` |
| `GDAL OK` ไม่ขึ้น | venv path ผิด | ตรวจ `PYTHONPATH` ใน container |
| Memory guard รอนาน | RAM เต็ม | ลด `N_JOBS` หรือเพิ่ม `MAX_RAM_GB` |
| `no imagery found` | path `/fs2` ไม่ได้ mount | ตรวจ volumes ใน compose / run.sh |
| Build นานมาก (>45 min) | compile GDAL ซ้ำ | ใช้ `docker build --cache-from` หรือ pull image จาก registry |