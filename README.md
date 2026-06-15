# NDVI Pipeline — ทุเรียน

คำนวณ NDVI รายเดือนจาก Sentinel-2 สำหรับ polygon พื้นที่ปลูกทุเรียน รันผ่าน Docker

## ภาพรวม workflow

```
polygon input (.parquet / .gpkg)
        │
        ▼
[Step 2] docker compose build        ← build image ครั้งเดียว
        │
        ▼
[Step 3] grid_point.py (ใน Docker)   ← สร้าง grid points 10m
        │
        ▼
data/grid_points/<PROVINCE>.parquet
        │
        ▼
[Step 4] ตั้งค่า env file
        │
        ▼
[Step 5-6] ndvi_pipeline.py (ใน Docker)
        │
        ▼
output/<province>/
  ├── ndvi_mean.parquet
  ├── ndvi_max.parquet
  ├── ndvi_min.parquet
  └── ndvi_median.parquet
```

---

## สิ่งที่ต้องมีก่อนเริ่ม

| สิ่งที่ต้องการ | หมายเหตุ |
|---|---|
| Docker Engine ≥ 24 + Docker Compose plugin | `docker compose` (ไม่ใช่ `docker-compose`) |
| ไฟล์ polygon ของพื้นที่ (.parquet หรือ .gpkg) | geometry column เป็น Polygon/MultiPolygon |
| เข้าถึง `/fs2/sentinel2/tiles` หรือ `/fs7/sentinel2/tiles` | Sentinel-2 tile data |
| RAM ≥ 16 GB | แนะนำ ≥ 64 GB สำหรับ N_JOBS=8 |
| พื้นที่ disk ≥ 20 GB | สำหรับ Docker image |

---

## ขั้นตอน

### Step 1 — Clone repo

```bash
git clone https://github.com/wanwisa-j/ndvi-durian-pipeline ndvi-pipeline
cd ndvi-pipeline
```

โครงสร้างไฟล์หลัง clone:

```
ndvi-pipeline/
├── grid_point.py          ← สร้าง grid points จาก polygon
├── ndvi_pipeline.py       ← pipeline หลัก (รันใน Docker)
├── Dockerfile
├── docker-compose.yaml
├── run.sh
├── requirements.txt
├── geodata/
│   └── thailand_bbox.gpkg
└── envs/
    ├── chanthaburi.env    ← ตัวอย่าง env (ต้องแก้ path)
    └── rayong.env
```

---

### Step 2 — Build Docker image

ครั้งแรกจะนานประมาณ **15–30 นาที** (compile GDAL จาก source)

```bash
docker compose build
# หรือ
./run.sh build
```

ตรวจสอบ:

```bash
docker images | grep ndvi-pipeline
# ควรเห็น ndvi-pipeline:latest
```

---

### Step 3 — สร้าง grid points (รันครั้งเดียวต่อ polygon file)

`grid_point.py` สร้าง point grid 10m aligned กับ Sentinel-2 pixels สำหรับแต่ละ polygon
Docker image มี `geopandas`, `shapely`, `numpy` ครบ ไม่ต้องติดตั้งอะไรเพิ่มบน host

```bash
docker run --rm \
  -v $(pwd):/workspaces \
  -v /fs2:/workspaces/fs2 \
  ndvi-pipeline:latest \
  python3 /workspaces/grid_point.py \
    --polygon-dir /workspaces/fs2/mydata/durian_polygons \
    --grid-dir    /workspaces/fs2/mydata/durian_grid_points
```

Output จะเป็น `.parquet` ใน `--grid-dir` ชื่อตาม stem ของ input file (uppercase):

```
polygons/
  RAYONG_durian.parquet  →  grid_points/RAYONG_DURIAN.parquet
  trat.gpkg              →  grid_points/TRAT.parquet
```

Pipeline จะ auto-discover grid file โดย match stem กับ `GOLDEN_DURIAN_PATH`
หรือตั้ง `GRID_POINTS_DIR` ใน env file ให้ชี้ตรงๆ ก็ได้

> รัน `--overwrite` เพื่อ rebuild ถ้า polygon เปลี่ยน

---

### Step 4 — เตรียม polygon file และสร้าง env file

ต้องการไฟล์ polygon ของพื้นที่ที่ต้องการคำนวณ NDVI

**รูปแบบที่รองรับ:**
- `.parquet` (GeoParquet)
- `.gpkg` (GeoPackage)

**ข้อกำหนด:**
- geometry column ต้องเป็น **Polygon** หรือ **MultiPolygon**
- มี column `plot_id` ถ้าไม่มีจะ auto-generate เป็น sequential 0, 1, 2, ...
- CRS ใดก็ได้ script จะแปลงเป็น EPSG:32647 อัตโนมัติ

สร้าง env file สำหรับแต่ละจังหวัด:

```bash
cp envs/chanthaburi.env envs/<province>.env
```

แก้ไข 3 ค่าที่ **จำเป็น**:

```env
PROVINCE=rayong
GOLDEN_DURIAN_PATH=/fs2/mydata/durian_polygons/RAYONG_durian.parquet
OUTPUT_DIR=/fs2/mydata/ndvi_output
```

ดู [ตาราง env vars ทั้งหมด](#env-vars) ด้านล่าง

> **path ใน `envs/*.env` hardcode ไว้กับเครื่องนี้** — ต้องแก้ให้ตรงกับ path ของคุณเสมอ

---

### Step 5 — รัน Pipeline

```bash
./run.sh run <province>

# ตัวอย่าง
./run.sh run rayong
./run.sh run chanthaburi
```

หรือใช้ docker compose โดยตรง:

```bash
docker compose --env-file envs/rayong.env up -d
docker compose --env-file envs/rayong.env logs -f
```

---

### Step 6 — ติดตาม progress

```bash
# ดู log real-time
./run.sh tail <province>

# ดูสถานะ container ทั้งหมด
./run.sh status

# เปิด shell เข้าใน container (debug)
./run.sh shell <province>

# หยุด container
./run.sh stop <province>
```

ทดสอบ GDAL ใน container:

```bash
./run.sh shell rayong
python3 -c "from osgeo import gdal; print('GDAL OK:', gdal.VersionInfo())"
env | grep -E "PROVINCE|GOLDEN|OUTPUT|S2_FOLDER"
```

---

## โครงสร้าง Output

```
$OUTPUT_DIR/
└── <province>/
    ├── point_master.parquet     ← grid point ทั้งหมด (WGS84)
    ├── ndvi_mean.parquet        ← NDVI เฉลี่ยรายเดือน
    ├── ndvi_max.parquet
    ├── ndvi_min.parquet
    ├── ndvi_median.parquet
    ├── performance_summary.json ← สถิติ runtime
    └── logs/
        └── ndvi_<province>_<YYYYMMDD_HHMMSS>.log
```

**Columns ใน `ndvi_*.parquet`:**

| Column | คำอธิบาย |
|--------|----------|
| `plot_id` | index ของ polygon ใน input (0-based) |
| `point_id` | index ของ point ภายใน polygon |
| `lat`, `lon` | พิกัด WGS84 |
| `geometry` | Point geometry (WGS84) |
| `YYYY-MM` | NDVI monthly value (float32, `NaN` = ไม่มีภาพใสพอในเดือนนั้น) |

---

## Env vars

| Variable | Default | Required | คำอธิบาย |
|---|---|---|---|
| `PROVINCE` | — | ✓ | ชื่อจังหวัด (lowercase) |
| `GOLDEN_DURIAN_PATH` | — | ✓ | path ของ polygon input |
| `OUTPUT_DIR` | — | ✓ | directory สำหรับ output |
| `GRID_POINTS_DIR` | auto | — | path ของโฟลเดอร์ grid parquet (auto-discover ถ้าไม่ระบุ) |
| `SENTINEL2_TILE_PATH` | `/fs2/angkanap/.../sentinel_2_index_shapefile.shp` | — | S2 tile grid shapefile |
| `S2_FOLDER_NEW` | `/fs2/sentinel2/tiles` | — | Sentinel-2 scenes หลัง 2025-07-01 |
| `S2_FOLDER_OLD` | `/fs7/sentinel2/tiles` | — | Sentinel-2 scenes ก่อน 2025-07-01 |
| `S2_FOLDER_CUTOFF` | `2025-07-01` | — | วันตัดระหว่าง OLD/NEW folder |
| `YEAR_START` | `2019` | — | ปีเริ่มต้น |
| `YEAR_END` | `2026` | — | ปีสิ้นสุด |
| `N_JOBS` | `auto` | — | จำนวน parallel worker (`auto` = คำนวณจาก RAM อัตโนมัติ) |
| `MAX_RAM_GB` | `100` | — | RAM budget (GB) |
| `FORCE_REBUILD_POINTS` | `0` | — | ตั้งเป็น `1` เพื่อ rebuild `point_master.parquet` |

---

## Troubleshooting

| อาการ | สาเหตุ | วิธีแก้ |
|---|---|---|
| `PROVINCE GRID NOT FOUND` | ยังไม่ได้รัน grid_point.py หรือ output path ผิด | รัน Step 3 และตั้ง `GRID_POINTS_DIR` ใน env ให้ชี้ตรงๆ |
| `'PROVINCE' is required` | ไม่ได้ส่ง env file | ใช้ `./run.sh run <province>` หรือ `--env-file envs/<province>.env` |
| `GDAL OK` ไม่ขึ้น | venv path ผิดใน container | ตรวจ `PYTHONPATH` ใน Dockerfile |
| Memory guard รอนานมาก | RAM ไม่เพียงพอ | ลด `N_JOBS` หรือเพิ่ม `MAX_RAM_GB` |
| `no imagery found` | S2 path ผิด หรือ `/fs` ไม่ได้ mount | ตรวจ `S2_FOLDER_NEW`/`OLD` และ volumes ใน docker-compose.yaml |
| Build นานมาก (>45 min) | compile GDAL ซ้ำ | ใช้ `docker build --cache-from` หรือ pull image จาก registry |
| output ไม่ครบ 4 ไฟล์ | pipeline exit ก่อนครบ | ดู log ด้วย `./run.sh tail <province>` หาบรรทัด `ERROR` |
