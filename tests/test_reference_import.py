"""
Offline tests for importing human reference boxes from Make Sense exports (Pascal VOC ZIP + CSV).

    python -m pytest tests -q
"""

import csv
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_consistency as ev  # noqa: E402
import import_reference as ir  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GT = REPO / "human_review" / "ground_truth"
REAL_ZIP = GT / "labels_my-project-name_2026-10-30-06-05-31.zip"
REAL_CSV = GT / "labels_my-project-name_2026-10-30-06-06-49.csv"


def quiet(*_a, **_k):
    pass


def voc_xml(name, w, h, boxes):
    objs = "".join(f"<object><name>{lab}</name><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin>"
                   f"<xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>" for lab, b in boxes)
    return f"<annotation><filename>{name}</filename><size><width>{w}</width><height>{h}</height></size>{objs}</annotation>"


class ImportTest(unittest.TestCase):
    W, H = 400, 300
    BOXES = [("TREE", (10, 20, 60, 80)), ("TREE", (100, 100, 150, 170)), ("TREE", (300, 200, 390, 290))]

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "imgs").mkdir()
        Image.new("RGB", (self.W, self.H), (30, 120, 40)).save(self.tmp / "imgs" / "SITE_27m.JPG", quality=90)
        Image.new("RGB", (self.W, self.H), (30, 90, 40)).save(self.tmp / "imgs" / "OTHER.JPG", quality=90)
        self.manifest = self.tmp / "manifest.csv"
        with self.manifest.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "image_path", "sha256"])
            w.writerow(["IMG_A", "imgs/SITE_27m.JPG", ""])
            w.writerow(["IMG_B", "imgs/OTHER.JPG", ""])
        self.review = self.tmp / "human_review"
        self.review.mkdir()
        ev.write_csv(self.review / "reference_boxes.csv", ev.REFERENCE_FIELDS,
                     [{"reference_id": "X1", "image_id": "IMG_B", "x_min_px": 1, "y_min_px": 1, "x_max_px": 9,
                       "y_max_px": 9, "label": "TREE", "annotator": "someone", "annotation_date": "", "notes": "keep"}])
        ev.write_csv(self.review / "reference_status.csv", ev.STATUS_FIELDS,
                     [{"image_id": "IMG_A", "image_sha256": "", "reference_status": "NONE"}])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def exports(self, voc_boxes=None, csv_boxes=None, name="SITE_27m.JPG", size=None, csv_size=None):
        voc_boxes = self.BOXES if voc_boxes is None else voc_boxes
        csv_boxes = voc_boxes if csv_boxes is None else csv_boxes
        w, h = size or (self.W, self.H)
        zp = self.tmp / "labels.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("SITE_27m.xml", voc_xml(name, w, h, voc_boxes))
        cp = self.tmp / "labels.csv"
        cw, ch = csv_size or (w, h)
        with cp.open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(ir.CSV_FIELDS)
            for lab, b in csv_boxes:
                wr.writerow([lab, b[0], b[1], b[2] - b[0], b[3] - b[1], name, cw, ch])
        return zp, cp

    def run_import(self, zp, cp, **kw):
        return ir.import_reference("IMG_A", zp, cp, "Annotator", "2026-10-30", self.review, self.manifest,
                                   self.tmp, log=quiet, **kw)

    def refs(self):
        return ev.read_csv(self.review / "reference_boxes.csv")

    def assert_refused(self, zp, cp, text, **kw):
        before = (self.review / "reference_boxes.csv").read_bytes()
        with self.assertRaises(ir.ImportError_) as cm:
            self.run_import(zp, cp, **kw)
        self.assertIn(text, str(cm.exception))
        self.assertEqual((self.review / "reference_boxes.csv").read_bytes(), before)

    def test_agreeing_exports_imported_unchanged(self):
        zp, cp = self.exports()
        zip_bytes, csv_bytes = zp.read_bytes(), cp.read_bytes()
        status_before = (self.review / "reference_status.csv").read_bytes()
        res = self.run_import(zp, cp)
        rows = self.refs()
        self.assertEqual([r["reference_id"] for r in rows], ["X1", "H01", "H02", "H03"])  # other image kept
        new = rows[1:]
        self.assertEqual([tuple(int(r[c]) for c in ("x_min_px", "y_min_px", "x_max_px", "y_max_px")) for r in new],
                         [b for _, b in self.BOXES])
        self.assertTrue(all(r["label"] == "TREE" and r["image_id"] == "IMG_A" for r in new))
        self.assertIn("VOC = CSV", new[0]["notes"])
        self.assertEqual((res["n_xml"], res["n_csv"]), (3, 3))
        self.assertEqual((zp.read_bytes(), cp.read_bytes()), (zip_bytes, csv_bytes))           # exports untouched
        self.assertEqual((self.review / "reference_status.csv").read_bytes(), status_before)   # not marked complete

    def test_lowercase_label_accepted(self):
        zp, cp = self.exports([("tree", b) for _, b in self.BOXES])
        self.run_import(zp, cp)
        self.assertEqual({r["label"] for r in self.refs()}, {"TREE"})

    def test_dry_run_writes_nothing(self):
        zp, cp = self.exports()
        before = (self.review / "reference_boxes.csv").read_bytes()
        res = self.run_import(zp, cp, dry_run=True)
        self.assertEqual(len(res["rows"]), 3)
        self.assertEqual((self.review / "reference_boxes.csv").read_bytes(), before)

    def test_one_pixel_rounding_needs_explicit_choice(self):
        csv_boxes = [self.BOXES[0], ("TREE", (100, 100, 151, 169)), self.BOXES[2]]
        zp, cp = self.exports(csv_boxes=csv_boxes)
        self.assert_refused(zp, cp, "Choose a source explicitly")
        self.run_import(zp, cp, use="csv")
        h02 = self.refs()[2]
        self.assertEqual([int(h02[c]) for c in ("x_min_px", "y_min_px", "x_max_px", "y_max_px")], [100, 100, 151, 169])
        self.assertIn("CSV used by explicit choice", h02["notes"])

    def test_explicit_choice_of_voc(self):
        zp, cp = self.exports(csv_boxes=[self.BOXES[0], ("TREE", (100, 100, 151, 169)), self.BOXES[2]])
        self.run_import(zp, cp, use="voc")
        h02 = self.refs()[2]
        self.assertEqual([int(h02[c]) for c in ("x_min_px", "y_min_px", "x_max_px", "y_max_px")], [100, 100, 150, 170])

    def test_larger_disagreement_refused_even_with_choice(self):
        zp, cp = self.exports(csv_boxes=[self.BOXES[0], ("TREE", (101, 100, 150, 170)), self.BOXES[2]])
        self.assert_refused(zp, cp, "exports disagree at object 2", use="voc")
        zp, cp = self.exports(csv_boxes=[self.BOXES[0], ("TREE", (100, 100, 152, 170)), self.BOXES[2]])
        self.assert_refused(zp, cp, "exports disagree at object 2", use="voc")

    def test_count_mismatch_refused(self):
        zp, cp = self.exports(csv_boxes=self.BOXES[:2])
        self.assert_refused(zp, cp, "3 XML boxes vs 2 CSV boxes")

    def test_wrong_label_refused(self):
        boxes = [self.BOXES[0], ("SHRUB", self.BOXES[1][1]), self.BOXES[2]]
        zp, cp = self.exports(boxes)
        self.assert_refused(zp, cp, "label 'SHRUB'")

    def test_out_of_bounds_and_inverted_refused(self):
        zp, cp = self.exports([self.BOXES[0], ("TREE", (350, 250, 410, 290))])
        self.assert_refused(zp, cp, "outside the image")
        zp, cp = self.exports([("TREE", (60, 20, 10, 80))], csv_boxes=[("TREE", (10, 20, 60, 80))])
        self.assert_refused(zp, cp, "outside the image or min >= max")

    def test_size_mismatch_refused(self):
        zp, cp = self.exports(size=(8192, 6144))
        self.assert_refused(zp, cp, "image size 8192x6144 != source image 400x300")
        zp, cp = self.exports(csv_size=(401, 300))
        self.assert_refused(zp, cp, "image size 401x300")

    def test_duplicate_box_refused(self):
        zp, cp = self.exports([self.BOXES[0], self.BOXES[0]])
        self.assert_refused(zp, cp, "duplicate boxes")

    def test_non_integer_coordinates_not_rounded(self):
        zp, _ = self.exports()
        cp = self.tmp / "labels.csv"
        cp.write_text(",".join(ir.CSV_FIELDS) + "\nTREE,10.5,20,50,60,SITE_27m.JPG,400,300\n")
        self.assert_refused(zp, cp, "not a whole pixel")

    def test_wrong_image_name_refused(self):
        zp, cp = self.exports(name="SITE_28m.JPG")
        self.assert_refused(zp, cp, "expected one XML for SITE_27m.JPG")

    def test_existing_rows_for_image_not_replaced(self):
        zp, cp = self.exports()
        self.run_import(zp, cp)
        self.assert_refused(zp, cp, "already has reference rows for IMG_A")

    def test_reference_images_show_only_human_boxes(self):
        import evaluation_plots as ep
        zp, cp = self.exports()
        res = self.run_import(zp, cp)
        paths = ep.reference_images(res["image_path"], "IMG_A", res["rows"], self.tmp / "out", "not approved (draft)")
        self.assertEqual([p.name for p in paths], ["human_reference_IMG_A.jpg", "human_reference_contact_sheet_IMG_A.png"])
        with Image.open(paths[0]) as im:
            self.assertEqual(im.width, self.W)          # full resolution, legend band below
            self.assertGreater(im.height, self.H)


@unittest.skipUnless(REAL_ZIP.exists() and REAL_CSV.exists(), "Make Sense exports not present")
class TestRealExports(unittest.TestCase):
    def test_pt_r1_exports(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            review = tmp / "review"
            review.mkdir()
            ev.write_csv(review / "reference_boxes.csv", ev.REFERENCE_FIELDS, [])
            args = ("LA_PT-R1_27m", REAL_ZIP, REAL_CSV, "Gabe", "2026-10-30", review, ev.MANIFEST, REPO)
            with self.assertRaises(ir.ImportError_):            # 1 px differences need an explicit choice
                ir.import_reference(*args, dry_run=True, log=quiet)
            res = ir.import_reference(*args, dry_run=True, use="voc", log=quiet)
            self.assertEqual((res["n_xml"], res["n_csv"], len(res["rows"])), (11, 11, 11))
            self.assertEqual((res["width"], res["height"]), (8192, 6144))
            self.assertEqual(len(res["rounding"]), 5)
            self.assertEqual(ev.read_csv(review / "reference_boxes.csv"), [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
