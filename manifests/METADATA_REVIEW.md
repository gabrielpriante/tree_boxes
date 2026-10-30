# Manifest fields awaiting your confirmation

Generated 2026-10-30 from `manifests/manifest.csv`. List the current state any time with
`python make_manifest.py --review`.

**How to confirm a field:** edit the value in `manifest.csv` if needed, then remove the field
name from that row's `needs_review` column. The runner will not send an image while any of
`site`, `week` or `condition` is UNKNOWN or still listed in `needs_review`, because those
values are inserted into the prompt. `coverage_scope` does not enter the prompt, so it does
not block a run, but it is needed for analysis.

Vocabulary: `UNKNOWN` = not yet established. `NOT_APPLICABLE` = does not apply.
`coverage_scope` must be `FULL_SITE`, `PARTIAL_SITE` or `UNKNOWN`.

## Priority 1: the ten October 24 altitude-series images

| image_id | rel. altitude (m) | week | condition | coverage_scope | other |
|---|---|---|---|---|---|
| LA_PT-02_64m | 64.3 | NOT_APPLICABLE | ALTITUDE_TEST | UNKNOWN | |
| LA_PT-02_66m | 66.2 | NOT_APPLICABLE | ALTITUDE_TEST | UNKNOWN | |
| LA_PT-02_67m | 67.2 | NOT_APPLICABLE | ALTITUDE_TEST | UNKNOWN | |
| LA_PT-02_71m | 71.2 | NOT_APPLICABLE | ALTITUDE_TEST | UNKNOWN | |
| LA_PT-02_86m | 85.8 | NOT_APPLICABLE | ALTITUDE_TEST | UNKNOWN | **heading rotated ~91°** |
| LA_PT-R1_27m | 26.8 | NOT_APPLICABLE | ALTITUDE_TEST | UNKNOWN | |
| LA_PT-R1_28m | 27.8 | NOT_APPLICABLE | ALTITUDE_TEST | UNKNOWN | |
| LA_PT-R1_30m | 30.2 | NOT_APPLICABLE | ALTITUDE_TEST | UNKNOWN | |
| LA_PT-R1_30m_angled | 30.1 | NOT_APPLICABLE | ALTITUDE_TEST | UNKNOWN | **heading rotated ~90°** |
| LA_PT-R1_31m | 31.2 | NOT_APPLICABLE | ALTITUDE_TEST | UNKNOWN | |

All were captured 2026-10-24 between 09:51 and 10:12 (EXIF). Altitudes are DJI relative
altitude (metres above the takeoff point), not height above the canopy.

Decisions needed:

1. **week / condition: confirmed 2026-10-30** as `NOT_APPLICABLE` / `ALTITUDE_TEST` for all ten.
   These strings are inserted into the prompt as image identifiers and state nothing about
   intervention timing.
2. **coverage_scope.** For each image, does the frame show the full site (`FULL_SITE`) or only
   part of it (`PARTIAL_SITE`)? This needs a visual check against the site boundary. It can't be
   derived from altitude or metadata. Not required before running; it matters only for analysis.
3. **The two rotated frames.** Both have nadir gimbal pitch (−89.9° / −90.0°), so neither is
   oblique. Their gimbal yaw differs by about 90° from the rest of their site's series:
   * `LA_PT-R1_30m_angled`: yaw +153.7° vs −116.4° for the other PT-R1 frames. The filename's
     "angled" matches this rotated heading, not a tilted camera. It duplicates the ~30 m step,
     so it is a heading/orientation variant, not an additional altitude.
   * `LA_PT-02_86m`: yaw −116.3° vs ≈ +152° for the other PT-02 frames. It is also the
     highest PT-02 frame, so altitude and heading change together in this image.

   Options: keep both in the altitude series and note the rotation in analysis; analyse them
   separately as orientation variants; or exclude them from the altitude comparison. To record
   your choice, edit `notes`, then remove `capture_geometry` from `needs_review`.

## Priority 2: week 1 condition labels (4 images)

`W01_MORNING-START_PT-02`, `W01_MORNING-START_PT-R1`, `W01_AFTERNOON-END_PT-02`,
`W01_AFTERNOON-END_PT-R1` keep their folder labels (`MORNING_START`, `AFTERNOON_END`).
Confirm whether these correspond to `BEFORE` / `AFTER` or should stay as their own documented
conditions. These four are blocked from running until confirmed.

## Priority 3: coverage_scope for the 16 week 3–6 images

`coverage_scope` is `UNKNOWN` for every standardized image. These images are not blocked.

## Not in the manifest

`PT_02.JPG` in the repository root is byte-identical to
`drone_images/week1/afternoon_end/PT_02.JPG` and is intentionally excluded.
