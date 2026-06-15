# NDVI Pipeline — Design Decisions & Code Walkthrough

> อธิบาย **ว่าโค้ดทำอะไร**, **ทำไมถึงออกแบบแบบนี้**, และ **trade-off ที่ตัดสินใจไป**
> สำหรับคนที่จะ maintain หรือ extend pipeline นี้ต่อ

---

## ภาพรวม Pipeline

```
Input Parquet (durian polygons)
        │
        ▼
Spatial Join ── S2 Tile Grid (.shp)
        │
        ▼
Per-tile loop
  │
  ├─ Discover scenes (band08, band04, OMNI)
  │
  └─ ProcessPoolExecutor (N workers)
        │
        ▼
     Per-polygon worker
        ├─ Grid 10m points inside polygon
        ├─ For each scene date:
        │    ├─ OMNI mask (clear pixels only)
        │    ├─ Windowed read band08 + band04
        │    ├─ DN-1000 correction (post 2022-01-25)
        │    └─ NDVI = (NIR - Red) / (NIR + Red)
        └─ Save → GeoParquet (1 file per polygon per tile)
```

---

## Configuration Layer

### ฟังก์ชัน `_require_env` / `_env`

```python
province           = _require_env("PROVINCE")
golden_durian_path = _require_env("GOLDEN_DURIAN_PATH")
output_dir         = _require_env("OUTPUT_DIR")
```

**ทำไมถึงแยก required กับ optional?**

Pipeline นี้รันหลายจังหวัดพร้อมกันบน server เดียว (process แยกกัน) ถ้า hardcode path จะต้องแก้โค้ดทุกครั้งที่เปลี่ยนจังหวัด แทนที่จะส่ง env file ต่างหาก

`_require_env` จงใจ raise ชัดเจนแทน silent default เพราะถ้า `PROVINCE` หาย แล้วให้ fallback เป็นค่าว่าง output จะเขียนผิด directory โดยไม่มีใครรู้ — **fail fast ดีกว่า fail silent**

---

## Logging

### ฟังก์ชัน `setup_logging` / `_worker_setup_logging`

**ปัญหาที่แก้:** `ProcessPoolExecutor` spawn worker processes แยก — ถ้าไม่ทำอะไร worker จะไม่มี handler เลย log หายหมด

**วิธีแก้:**
- `setup_logging()` เรียกครั้งเดียวใน main process → สร้างไฟล์ log พร้อม timestamp
- path ของ log file ส่งผ่าน argument ไปให้ทุก worker
- `_worker_setup_logging(log_path)` เรียกใน **ต้น** ของ `process_polygon` → แต่ละ worker attach ไปที่ไฟล์เดิม (append mode)
- มี guard `if root.handlers: return` กันการ attach ซ้ำถ้า worker รับงานหลาย polygon

**ทำไมไม่ใช้ `logging.handlers.QueueHandler`?**
Queue-based logging ซับซ้อนกว่าและต้องมี listener process แยก สำหรับ workload นี้ (I/O bound, log ไม่ได้ dense มาก) การ append พร้อมกันหลาย process ไปยังไฟล์เดียวบน Linux ปลอดภัยพอ (write < 4KB atomic บน ext4)

---

## Memory Management

### ฟังก์ชัน `_get_system_memory_gb` / `_get_process_tree_memory_gb` / `wait_for_memory_budget`

**ปัญหาที่แก้:** NDVI extraction โหลด raster ขนาดใหญ่ในหลาย worker พร้อมกัน ถ้า submit งานเร็วเกินไป RAM จะเต็มก่อนที่ worker เก่าจะ GC

**Metric priority: PSS > USS > RSS**

| Metric | ความหมาย | ปัญหา |
|--------|-----------|-------|
| RSS | memory ที่ process จอง รวม shared pages | inflate เพราะนับ shared lib ซ้ำ |
| USS | memory ที่ process ใช้จริง ไม่นับ shared | ต้องการ `/proc/smaps` (อาจ permission deny) |
| PSS | USS + สัดส่วนของ shared pages | แม่นที่สุด แต่ก็ต้องการ smaps |

โค้ดลอง PSS ก่อน ถ้าไม่ได้ลอง USS แล้ว fallback RSS — เพื่อให้ทำงานได้แม้ใน container ที่ lock `/proc`

**RSS special case:**
```python
if metric == "rss":
    over_budget = tree_gb >= threshold and sys_avail < min_free_ram_gb
```
RSS บวมเสมอเพราะ CoW pages หลัง `fork()` — ถ้า block ที่ RSS อย่างเดียวจะ false-positive เกือบตลอด จึงต้องมี `sys_avail` เป็นเงื่อนไขร่วมด้วย

**`wait_for_memory_budget` logic:**
```
while RAM ตึง:
    รอ POLL_SECONDS วินาที
    ถ้าเกิน MAX_WAIT_SECONDS แต่ sys_avail ยังพอ → ปล่อยผ่าน (warning)
```
ไม่ raise error เพราะ pipeline ยาว หยุดกลางคันเสียหายกว่า — แค่ log warning แล้วยอมให้รัน

### ฟังก์ชัน `cleanup_memory`

```python
def cleanup_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
```

Python GC คืน memory กลับไปที่ Python allocator แต่ allocator ไม่ได้คืน OS เสมอไป `malloc_trim(0)` บอก glibc ให้คืน heap ส่วนที่ว่างกลับ OS จริงๆ สำคัญมากสำหรับ long-running process ที่ allocate/free รอบใหญ่ซ้ำๆ

---

## Scene Discovery

### ฟังก์ชัน `s2_timeseries_fullpath`

**ทำไมไม่ใช้ glob แบบ recursive?**

Directory structure ของ Sentinel-2 ที่ `/fs2/sentinel2/tiles` ถูก organize ตาม MGRS tile:
```
tiles/
└── 47/P/PS/
    └── S2B_MSIL2A_20230115_...
        └── GRANULE/.../IMG_DATA/R10m/*_B08_10m.jp2
```

`os.listdir()` + filter ด้วย string operations เร็วกว่า `glob(**/)` มากบน NFS เพราะลด syscall และไม่ต้อง traverse directory ที่ไม่เกี่ยว

**การ validate:**
- ตรวจ `parts[2][0:4]` เป็น year ที่ต้องการ → กรอง scene นอก range ออกไวโดยไม่ต้อง stat ทุกไฟล์
- ตรวจ `fn[2:3] == sat` → กรองว่าเป็น Sentinel-2A หรือ 2B (SAT env var)
- ต้องมี OMNI SCL file และ band file **ทั้งคู่** ถึงจะ include — ป้องกัน scene ที่ดาวน์โหลดไม่ครบ

**Output:** DataFrame เรียงตาม date พร้อม path ครบทั้ง band08, band04, omni

---

## Point Grid Generation

### ฟังก์ชัน `create_points_in_polygon`

```python
xx, yy = np.meshgrid(
    np.arange(minx, maxx, spacing),
    np.arange(miny, maxy, spacing),
)
mask = contains(geom, xx.ravel(), yy.ravel())
```

**ทำไม meshgrid แทน loop?**

Polygon ขนาด 1 เฮกตาร์ = ~100 points บน grid 10m, แต่ polygon ใหญ่อาจมีหลักหมื่น NumPy meshgrid + vectorized `contains` เร็วกว่า Python loop หลาย order of magnitude

**ทำไม spacing=10 fixed?**

Pipeline ออกแบบสำหรับ Sentinel-2 band 10m เท่านั้น (B08, B04) การใส่ spacing ที่ไม่ใช่ 10 จะ sample ระหว่าง pixel ซึ่งไม่มีประโยชน์ จึง hardcode ไว้ที่ `res = 10` ระดับ config

---

## Windowed Raster Read

### ฟังก์ชัน `sample_raster_windowed`

**ปัญหาเดิม (ถ้าอ่าน full tile):**
Sentinel-2 tile ขนาด 110×110 km ที่ 10m resolution = 11,000×11,000 pixels ≈ **484 MB per band** ถ้าอ่านทั้ง tile ทุก scene ทุก band RAM จะหมดใน worker แรก

**วิธีแก้ — windowed read:**
```python
px = ((xs - gt[0]) / gt[1]).astype(np.int32)   # แปลง coord → pixel index
py = ((ys - gt[3]) / gt[5]).astype(np.int32)

xmin, xmax = px.min(), px.max()
ymin, ymax = py.min(), py.max()

arr = band.ReadAsArray(xmin, ymin, width, height)  # อ่านแค่ bounding box
```

สำหรับ polygon เล็ก (~1 เฮกตาร์) window ที่อ่านจริงอาจแค่ 10×10 pixels = **400 bytes** แทน 484 MB — ประหยัดได้มหาศาล

**Clamp ก่อน:**
```python
px = np.clip(px, 0, ds.RasterXSize - 1)
```
ป้องกัน pixel ที่อยู่บริเวณขอบ polygon หลุดออกนอก raster extent (floating point rounding)

---

## DN Baseline Correction

### ฟังก์ชัน `apply_dn_correction`

**Background:**
ESA เปลี่ยน Sentinel-2 L2A processing baseline เป็น 04.00 เมื่อ **25 มกราคม 2022** โดยเพิ่ม offset +1000 เข้าไปใน DN ทุก band เพื่อรองรับค่าติดลบ (surface reflectance บางพื้นที่เป็นลบได้)

**ผลกระทบต่อ NDVI:**
ถ้าไม่แก้ไข ภาพก่อนและหลัง 2022-01-25 จะมี NDVI ต่างกันอย่างเป็นระบบแม้พืชเดิม เพราะ:
```
NDVI_before = (NIR - Red) / (NIR + Red)          # ค่าจริง
NDVI_after  = ((NIR+1000) - (Red+1000)) / (...)  # offset หักล้างกันในตัวเศษ
                                                   # แต่ตัวส่วนต่างออกไป → NDVI เพี้ยน
```

**วิธีแก้:**
```python
DN_CORRECTION_CUTOFF = pd.Timestamp("2022-01-25")

if acq_date > DN_CORRECTION_CUTOFF:
    return values - DN_CORRECTION_VALUE  # ลบ 1000 ก่อนคำนวณ NDVI
```

ใช้ `>` ไม่ใช่ `>=` เพราะ baseline 04.00 apply กับภาพที่ process **หลังจาก** วันนั้น ภาพที่ถ่ายวันเดียวกันแต่ process ก่อนอาจยังเป็น baseline เก่า

---

## Parallel Processing

### `ProcessPoolExecutor` ใน `main()`

**ทำไม `ProcessPoolExecutor` ไม่ใช่ `ThreadPoolExecutor`?**

GDAL และ NumPy release GIL บางส่วนแต่ไม่ทั้งหมด — Python GIL จะกลายเป็น bottleneck ถ้าใช้ thread สำหรับ CPU-bound work แบบนี้ Process pool ให้ parallelism จริงๆ บน multi-core

**ทำไมไม่ใช้ `joblib.Parallel`?**
`joblib` มี memory backend ที่ซับซ้อนและ lifecycle ของ worker pool ควบคุมได้ยากกว่า `ProcessPoolExecutor` ซึ่ง context manager รับประกันว่า worker จะถูก shutdown เรียบร้อยเมื่อ tile เสร็จ ลด zombie process

**Sliding window submit pattern:**
```python
# Seed N_JOBS งานก่อน
while len(pending) < N_JOBS and not exhausted:
    _submit_next()

# เมื่อ future เสร็จ → submit งานใหม่แทนที่
while pending:
    for future in as_completed(...):
        pending.pop(future)
        ...
        _submit_next()   # เติมงานใหม่
        break            # re-enter as_completed ด้วย pending ที่อัพเดต
```

แทนที่จะ `pool.map()` ทุกอย่างพร้อมกัน pattern นี้รักษา **concurrency คงที่** (เสมอมีงาน N_JOBS ใน flight) โดยไม่ต้องโหลด task list ทั้งหมดใน memory และยัง hook `wait_for_memory_budget()` ก่อน submit แต่ละงานได้

**Memory guard integration:**
```python
def _submit_next():
    wait_for_memory_budget()   # block ถ้า RAM ตึง
    f = pool.submit(process_polygon, ...)
```

ถ้า RAM ตึงจะ pause การ submit งานใหม่ แต่ worker ที่รันอยู่จะรันต่อจนเสร็จ → ค่อยๆ free memory ก่อนรับงานใหม่

---

## Per-Polygon Worker

### ฟังก์ชัน `process_polygon`

**ทำไมต้องเป็น module-level function?**

`ProcessPoolExecutor` ใช้ `pickle` ส่ง task ไป worker — `pickle` serialize ได้เฉพาะ function ที่ defined ที่ module level ไม่ใช่ lambda หรือ nested function

**Skip if exists:**
```python
if os.path.exists(ndvi_out):
    return ndvi_out
```
Pipeline ออกแบบให้ **idempotent** — รันซ้ำได้โดยข้าม polygon ที่ทำแล้ว ช่วยมากเวลา pipeline crash กลางคันแล้วต้อง resume

**GDAL handle cleanup ใน `finally`:**
```python
finally:
    omni_ds = b1_ds = b2_ds = None
```
GDAL dataset ไม่ได้ใช้ context manager — ต้อง assign `None` เพื่อให้ destructor ปิด file handle GDAL เองถ้าไม่ปิดจะ leak file descriptor ใน long-running worker

**OMNI SCL mask:**
```python
ov = sample_raster_windowed(omni_ds, xs, ys)
clear_mask = ov == 0   # 0 = clear sky ใน SCL
```
Sentinel-2 Scene Classification Layer (SCL) ที่ resolution 20m (resampled มาเป็น OMNI) encode สภาพบรรยากาศ — `0` หมายถึง clear pixel เท่านั้นที่เชื่อถือได้สำหรับ NDVI cloud, shadow, snow ถูกกรองออกหมด

---

## Output Format

**ทำไม GeoParquet ไม่ใช่ CSV หรือ GeoTIFF?**

| Format | ข้อดี | ข้อเสีย |
|--------|-------|---------|
| CSV | อ่านง่าย | ไม่มี geometry, ไฟล์ใหญ่, โหลดช้า |
| GeoTIFF | เป็นมาตรฐาน raster | time-series หลาย band ซับซ้อน |
| GeoParquet | columnar, compressed, มี geometry | ต้องการ geopandas/pyarrow อ่าน |

Pipeline downstream ใช้ Python + geopandas อยู่แล้ว GeoParquet ให้ **columnar read** — ถ้าต้องการแค่ date บางวันไม่ต้อง load ทุก column

**1 file per polygon per tile:**
```
<province>_segment_ndvi_<tile>_<idx>.parquet
```
แยกไฟล์ย่อยแทน merge ใหญ่เพราะ:
- Resume ได้ (skip if exists)
- Worker เขียนได้พร้อมกันโดยไม่ lock กัน
- ถ้า polygon บางตัว corrupt ไม่กระทบตัวอื่น

---

## สิ่งที่ควร Migrate ต่อ

| จุด | สถานะปัจจุบัน | แนะนำ |
|-----|---------------|--------|
| `shapely.vectorized.contains` | legacy API, deprecation warning ใน shapely ≥2.0 | เปลี่ยนเป็น `shapely.contains_xy(geom, xs, ys)` |
| Log rotation | ไม่มี (append ไปเรื่อยๆ) | ใช้ `RotatingFileHandler` |
| OMNI SCL value | hardcode `== 0` | document ค่า SCL ทั้งหมดหรือ configurable |
| Output merge | ไม่มี | เพิ่ม post-processing step merge parquet per province |

---

## Session 2025-06-08 — Bug Fixes

### 1. Schema mismatch ใน `aggregate_ndvi` (🔴 Critical) — **แก้แล้ว**

**ปัญหา:**
`pq.ParquetWriter` ถูก init ด้วย schema ของไฟล์แรก แต่ไฟล์จาก tile ต่างกันอาจมี column เดือนต่างกัน
(เช่น tile A มี 2019-01..2022-12, tile B มีต่างออกไป) → `write_table` ครั้งที่ 2 จะ fail หรือ silently corrupt

**วิธีแก้:**
Two-pass approach:
- Pass 1: อ่านแค่ schema ของทุกไฟล์ด้วย `pq.read_schema()` (เร็ว) เพื่อ collect union ของ column เดือนทั้งหมด
- Pass 2: process แต่ละไฟล์ โดยถ้าเดือนไหนไม่มีในไฟล์นั้นให้ใส่ `np.nan` แทน
- สร้าง ParquetWriter ครั้งเดียวตอนไฟล์แรก schema จึงตรงกันทุกไฟล์

**ทำไมไม่ทำ single-pass?**
Schema ของไฟล์แรกไม่รู้ว่า tile อื่นมีเดือนอะไรบ้าง ต้อง scan ทุกไฟล์ก่อน

---

### 2. Memory guard infinite hang (🔴 Critical) — **แก้แล้ว**

**ปัญหา:**
Timeout check ใน `wait_for_memory_budget` (บรรทัด ~307):
```python
if waited >= max_wait_seconds and effective_avail >= min_free_ram_gb:
    return
```
ถ้า RAM ไม่ว่างเลย (`effective_avail < min_free_ram_gb`) loop จะวนตลอดไปแม้เกิน `max_wait_seconds` แล้ว

**วิธีแก้:**
เอา `and effective_avail >= min_free_ram_gb` ออก → return unconditionally หลัง timeout
ยัง log warning บอกว่าเป็น "forced continue" เพื่อ track ใน log

**Trade-off:**
ยอมเสี่ยง OOM เล็กน้อยเพื่อหลีก infinite hang ซึ่งแย่กว่า (pipeline ค้างตลอด)

---

### 3. `as_completed` + `break` สร้าง O(n²) overhead (🟠 สำคัญ) — **แก้แล้ว**

**ปัญหา:**
Pattern เดิม:
```python
while pending:
    for future in as_completed(list(pending.keys())):
        ...
        break   # สร้าง generator ใหม่ทุกรอบ
```
`as_completed` subscribe ไปยังทุก future ทุกครั้งที่สร้าง → O(n) per completion → O(n²) รวม

**วิธีแก้:**
ใช้ `concurrent.futures.wait(list(pending), return_when=FIRST_COMPLETED)` แทน
- return ทันทีที่มี future เสร็จ ≥1 ตัว
- process ทุกตัวที่เสร็จพร้อมกัน (อาจ >1 ถ้า finish ใกล้กัน)
- submit งานใหม่ทดแทนจำนวนที่เสร็จ

---

### จุดที่ตัดสินใจ **ยังไม่แก้** ในรอบนี้

| จุด | เหตุผล |
|-----|---------|
| DN correction — ค่าติดลบหลังลบ 1000 | ต้องตรวจสอบข้อมูลจริงก่อนว่ามีกรณีนี้เกิดขึ้นจริงไหม |
| OMNI cloud mask `ov == 0` อาจผิด | Decision.md ระบุว่า 0 = clear ใน OMNI SCL format นี้ ต้องยืนยันกับทีม remote sensing |
| N_JOBS คำนวณ ณ import time | minor, ไม่กระทบ production |
| total_submitted นับรวม skip | ปัญหา metric เท่านั้น ไม่กระทบ output |
---

## Session 2026-06-08 — Scene-first Architecture Rewrite

### Context

Rayong run (31,395 polygons, 1 tile 47PQQ, ~53 scenes) estimated 14 hours.
Bottleneck analysis:

| Source | Count |
|--------|-------|
| Polygons | 31,395 |
| Scenes | ~250 |
| Bands (OMNI + B08 + B04) | 3 |
| **Total GDAL opens (old)** | **~23 million** |
| **Total GDAL opens (new)** | **~750** |

User confirmed: "rewrite เลยดีกว่าใช่มั้ย"

---

### Architecture Change: Polygon-first → Scene-first

**Before** (`process_polygon`):
```
for tile:
  for polygon (parallel, N_JOBS workers):
    for scene:
      gdal.Open(SCL)  ← opened 31k × per worker
      gdal.Open(B08)
      gdal.Open(B04)
      read_window → NDVI
    save polygon_N.parquet  ← 31k files
```

**After** (`process_scene`):
```
for tile:
  build_or_load_points_cache()  ← computed once, cached to disk
  for scene (parallel, N_JOBS workers):
    gdal.Open(SCL)  ← opened ONCE per scene
    gdal.Open(B08)
    gdal.Open(B04)
    for all polygons: read_window → NDVI
  save tile_TILENAME.parquet  ← 1 file per tile (1-3 total)
```

---

### Key Design Decisions

**1. Worker initializer (`_worker_init`)**
- Point grid arrays (xs, ys, offsets) sent to each worker ONCE via `ProcessPoolExecutor(initializer=_worker_init, initargs=...)`
- Without this: 250 scenes × 8 workers × ~24 MB pickle = 48 GB of pickling overhead
- With initializer: 8 workers × 24 MB = 192 MB one-time cost

**2. Flat array layout**
- All polygon points stored in a single flat array: `xs_flat[offsets[i]:offsets[i+1]]` = polygon i
- Allows single numpy allocation, avoids per-polygon dict overhead
- `offsets` array = prefix sums of per-polygon point counts

**3. Point grid cache** (`_pts_cache_{tile}.parquet`)
- Computed once per tile, reused across reruns
- Cost = shapely `contains_xy` over meshgrid × n_polygons (minutes once, zero thereafter)
- Stored as: `poly_idx, plot_id, point_id, x, y`

**4. Tile-level output** (1 file per S2 tile)
- Replaces 31k per-polygon parquets → 1-3 tile parquets
- Eliminates: 31k file creates, opens, closes, globs, deletes
- Aggregate reads 1-3 large files instead of 31k small ones
- Row group size = 200k rows for efficient column-scan in aggregate

**5. Resume / idempotency**
- Tile output exists? Skip entire tile (no per-polygon granularity)
- Point cache exists? Load from disk (skip geometry computation)
- To rerun a tile: delete `{province}_{tree}_ndvi_tile_{TILE}.parquet`
- To regenerate point cache: delete `_pts_cache_{TILE}.parquet`

---

### Expected Runtime

| Phase | Before | After |
|-------|--------|-------|
| GDAL opens | 23M | ~750 |
| Per-polygon parquet I/O | 31k files | 0 |
| Tile-level save | — | 1-3 files |
| **Total estimated** | **~12-14h** | **~1.5-2.5h** |

Main speedup source: GDAL JP2 open = ~10ms × 23M = 64 hours possible worst-case.
Actual improvement depends on NFS cache warmth and JP2 tile layout.

---

### Trade-offs Accepted

| Trade-off | Decision |
|-----------|----------|
| Resume granularity coarser (per-tile not per-polygon) | Acceptable — tile takes 30-60 min not 14h |
| Peak memory higher (all scenes for tile in RAM) | ~1.5 GB per tile at 250 scenes × 1.5M pts — fine on 91GB server |
| `poly_idx` ordering must be stable in cache | `pts` sorted by poly_idx from `tile_polygons.iterrows()` — stable |

---

## Session 2026-06-08 (continued) — Output Structure Decisions

Confirmed via explicit user approval:

### 1. `point_master.parquet` — province-wide GeoParquet

**Schema:** `plot_id, point_id, x, y, lat, lon, geometry`
- `x` = **longitude** (WGS84)
- `y` = **latitude** (WGS84)
- `geometry` = Point in EPSG:4326
- Built by concatenating all `_pts_cache_{tile}.parquet` files
- Written to `{OUTPUT_DIR}/{PROVINCE}/point_master.parquet`
- Skipped if file already exists (unless `FORCE_REBUILD_POINTS=1`)

**Rationale:** x/y in WGS84 (not UTM) so downstream tools using `x, y` columns directly get geographic coords without a separate reprojection step.

### 2. `_pts_cache_{tile}.parquet` — kept between runs

- **Never deleted by the pipeline**
- Contains `poly_idx, plot_id, point_id, x(UTM), y(UTM), lat(WGS84), lon(WGS84)`
- lat/lon precomputed at cache-build time (not at sample time) — avoids per-scene reprojection
- To regenerate: set `FORCE_REBUILD_POINTS=1` or delete the file manually
- Location: `{OUTPUT_DIR}/{PROVINCE}/_pts_cache_{tile}.parquet`

**Rationale:** Point grid computation (shapely `contains_xy` over meshgrid) takes minutes for large polygon sets. Keeping the cache avoids this cost on every rerun.

### 3. `_ndvi_tile_{tree}_{tile}.parquet` — deleted after aggregate

- Intermediate file: holds flat NDVI values (lat, lon, date columns) for one tile
- Format: plain Parquet (no geometry) — geometry added during aggregate
- **Automatically deleted** after `aggregate_ndvi()` completes successfully
- If aggregate fails, intermediate files are preserved for debugging/resume

**Rationale:** These files can be 1-2 GB each. Keeping them after aggregate doubles storage with no benefit since the final `ndvi_mean/max/min/median.parquet` files are the canonical output.

---

## Session 2026-06-08 (continued 2) — Province Grid as Authoritative Point Source

### Discovery (from running pipeline)

During first Rayong run, the pipeline spent 30+ min generating 2.3M points via `create_points_in_polygon` (meshgrid + `contains_xy` for 31,395 polygons). Pre-generated grids already exist in `data/grid_points/` for all provinces.

### Decision: Use pre-generated grid, never regenerate silently

**Source:** `data/grid_points/{PROVINCE_STEM_UPPER}.parquet`
- Authoritative. Do NOT run `create_points_in_polygon` unless grid is absent.
- If grid not found → **exit with error** (never regenerate silently)
- `GRID_POINTS_DIR` env var overrides auto-discovery path

**Auto-discovery rule:**
```
GOLDEN_DURIAN_PATH = .../data/polygons/RAYONG_durian_v2_cuda_post_post.parquet
grid_dir = dirname(dirname(GOLDEN_DURIAN_PATH))/grid_points/
stem_match = stem.upper() with trailing " N" stripped
```

**Grid schema (verified):**
- `plot_id` = 0-based polygon row index in polygon file (not 1-indexed)
- `lon, lat` = WGS84 (used for reprojection to tile UTM — do NOT trust `x_utm/y_utm` for cross-zone tiles)
- Required columns: `plot_id, point_id, lon, lat`

**`_pts_cache_{tile}.parquet` is now obsolete** for the primary path.
The per-tile cache was only needed when point generation was expensive. Since the province grid loads in ~3s, per-tile caches add no value. Decision.md entries about `_pts_cache_*.parquet` remain valid only for hypothetical fallback runs.

### Performance impact

| Phase | Before | After |
|-------|--------|-------|
| Point preparation (Rayong) | ~30 min (generate 2.3M pts) | ~3s (load grid) + ~5s (filter+reproject per tile) |
| workers=0 window | ~30 min | ~10s |

### workers=0 explanation

`workers=0` in ResourceMonitor is expected. The `ProcessPoolExecutor` is created per-tile, so the monitor shows 0 workers during the startup and grid-loading phases. Not a bug.

