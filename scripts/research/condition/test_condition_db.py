#!/usr/bin/env python3
"""Validate the condition assessment DB schema: migration, CRUD, foreign keys.

Usage:
    python scripts/test_condition_db.py          # run all checks
    python scripts/test_condition_db.py --migrate # run migration first, then checks

Exits 0 on success, 1 on failure.
"""

import argparse
import sys
import traceback

from sqlalchemy import inspect, text

from cardprice.db.session import engine

# ---------------------------------------------------------------------------
# Expected schema definitions
# ---------------------------------------------------------------------------
CONDITION_TABLES = {
    "condition_scans": {
        "columns": {
            "scan_id": "integer",
            "card_id": "text",
            "inventory_id": "integer",
            "overall_grade": "real",
            "tcg_condition": "text",
            "confidence": "real",
            "grade_ci_low": "real",
            "grade_ci_high": "real",
            "model_version": "text",
            "raw_output": "jsonb",
            "created_at": "timestamp with time zone",
        },
        "pk": "scan_id",
        "fk_targets": ["dim_cards", "user_inventory"],
    },
    "condition_images": {
        "columns": {
            "image_id": "integer",
            "scan_id": "integer",
            "angle_type": "text",
            "image_path": "text",
            "image_quality": "real",
            "resolution_w": "integer",
            "resolution_h": "integer",
            "created_at": "timestamp with time zone",
        },
        "pk": "image_id",
        "fk_targets": ["condition_scans"],
        "cascade_delete": True,
    },
    "condition_scores": {
        "columns": {
            "score_id": "integer",
            "scan_id": "integer",
            "category": "text",
            "score": "real",
            "confidence": "real",
            "defects": "jsonb",
            "source_images": "ARRAY",
        },
        "pk": "score_id",
        "fk_targets": ["condition_scans"],
        "cascade_delete": True,
    },
    "condition_calibration": {
        "columns": {
            "id": "integer",
            "scan_id": "integer",
            "grade_authority": "text",
            "actual_grade": "real",
            "actual_subgrades": "jsonb",
            "cert_number": "text",
            "predicted_grade": "real",
        },
        "pk": "id",
        "fk_targets": ["condition_scans"],
    },
}

# Check-constraint values we expect
VALID_TCG_CONDITIONS = {"NM", "LP", "MP", "HP", "DMG"}
VALID_ANGLE_TYPES = {
    "front", "back", "oblique_front", "oblique_back",
    "edge_top", "edge_bottom", "edge_left", "edge_right",
    "corner_tl", "corner_tr", "corner_bl", "corner_br",
}
VALID_CATEGORIES = {"centering", "corners", "edges", "surface"}
VALID_GRADE_AUTHORITIES = {"PSA", "BGS", "CGC"}


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, msg: str):
        self.passed += 1
        print(f"  PASS: {msg}")

    def fail(self, msg: str):
        self.failed += 1
        self.errors.append(msg)
        print(f"  FAIL: {msg}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("\nFailures:")
            for e in self.errors:
                print(f"  - {e}")
        print(f"{'='*60}")
        return self.failed == 0


def check_tables_exist(inspector, result: TestResult):
    """Verify all four condition tables exist."""
    print("\n[1] Checking condition tables exist...")
    existing = set(inspector.get_table_names())
    for tbl in CONDITION_TABLES:
        if tbl in existing:
            result.ok(f"Table '{tbl}' exists")
        else:
            result.fail(f"Table '{tbl}' does NOT exist")


def check_columns(inspector, result: TestResult):
    """Verify each table has the expected columns with correct types."""
    print("\n[2] Checking column definitions...")
    for tbl, spec in CONDITION_TABLES.items():
        try:
            cols = {c["name"]: c for c in inspector.get_columns(tbl)}
        except Exception as e:
            result.fail(f"Cannot inspect columns of '{tbl}': {e}")
            continue

        for col_name, expected_type_fragment in spec["columns"].items():
            if col_name not in cols:
                result.fail(f"{tbl}.{col_name} missing")
                continue
            actual_type = str(cols[col_name]["type"]).lower()
            if expected_type_fragment.lower() in actual_type:
                result.ok(f"{tbl}.{col_name} type OK ({actual_type})")
            else:
                result.fail(
                    f"{tbl}.{col_name} type mismatch: "
                    f"expected ~'{expected_type_fragment}', got '{actual_type}'"
                )


def check_primary_keys(inspector, result: TestResult):
    """Verify primary keys."""
    print("\n[3] Checking primary keys...")
    for tbl, spec in CONDITION_TABLES.items():
        try:
            pk = inspector.get_pk_constraint(tbl)
            pk_cols = pk.get("constrained_columns", [])
        except Exception as e:
            result.fail(f"Cannot get PK for '{tbl}': {e}")
            continue

        if spec["pk"] in pk_cols:
            result.ok(f"{tbl} PK = {pk_cols}")
        else:
            result.fail(f"{tbl} PK expected '{spec['pk']}', got {pk_cols}")


def check_foreign_keys(inspector, result: TestResult):
    """Verify foreign key references."""
    print("\n[4] Checking foreign keys...")
    for tbl, spec in CONDITION_TABLES.items():
        try:
            fks = inspector.get_foreign_keys(tbl)
        except Exception as e:
            result.fail(f"Cannot get FKs for '{tbl}': {e}")
            continue

        fk_referred = {fk["referred_table"] for fk in fks}
        for target in spec["fk_targets"]:
            if target in fk_referred:
                result.ok(f"{tbl} -> {target} FK exists")
            else:
                result.fail(f"{tbl} -> {target} FK missing (found: {fk_referred})")


def check_indexes(inspector, result: TestResult):
    """Verify key indexes exist."""
    print("\n[5] Checking indexes...")
    expected_indexes = {
        "condition_scans": [
            "idx_cond_scans_card",
            "idx_cond_scans_inventory",
            "idx_cond_scans_grade",
        ],
        "condition_images": ["idx_cond_images_scan"],
        "condition_scores": ["idx_cond_scores_scan"],
        "condition_calibration": [
            "idx_cond_calib_scan",
            "idx_cond_calib_cert",
        ],
    }
    for tbl, idx_names in expected_indexes.items():
        try:
            indexes = {idx["name"] for idx in inspector.get_indexes(tbl)}
        except Exception as e:
            result.fail(f"Cannot get indexes for '{tbl}': {e}")
            continue

        for idx in idx_names:
            if idx in indexes:
                result.ok(f"{tbl} index '{idx}' exists")
            else:
                result.fail(f"{tbl} index '{idx}' missing (found: {indexes})")


def check_crud(result: TestResult):
    """Insert, read, update, delete test records across all condition tables."""
    print("\n[6] Testing CRUD operations...")

    with engine.connect() as conn:
        try:
            # We need a valid card_id for the FK. Pick one from dim_cards.
            card_id = conn.execute(
                text("SELECT card_id FROM dim_cards LIMIT 1")
            ).scalar()
            if not card_id:
                result.fail("No cards in dim_cards; cannot test FK insert")
                return

            result.ok(f"Found test card_id: {card_id}")

            # --- INSERT condition_scans ---
            scan_id = conn.execute(text("""
                INSERT INTO condition_scans
                    (card_id, overall_grade, tcg_condition, confidence,
                     grade_ci_low, grade_ci_high, model_version, raw_output)
                VALUES
                    (:card_id, 8.5, 'NM', 0.92, 8.0, 9.0, 'test_v1',
                     '{"test": true}'::jsonb)
                RETURNING scan_id
            """), {"card_id": card_id}).scalar()
            result.ok(f"INSERT condition_scans OK (scan_id={scan_id})")

            # --- INSERT condition_images ---
            image_id = conn.execute(text("""
                INSERT INTO condition_images
                    (scan_id, angle_type, image_path, image_quality,
                     resolution_w, resolution_h)
                VALUES
                    (:scan_id, 'front', '/tmp/test_front.jpg', 0.95, 1920, 2560)
                RETURNING image_id
            """), {"scan_id": scan_id}).scalar()
            result.ok(f"INSERT condition_images OK (image_id={image_id})")

            # --- INSERT condition_scores (all 4 categories) ---
            for cat in VALID_CATEGORIES:
                conn.execute(text("""
                    INSERT INTO condition_scores
                        (scan_id, category, score, confidence, defects)
                    VALUES
                        (:scan_id, :cat, :score, 0.88,
                         :defects::jsonb)
                """), {
                    "scan_id": scan_id,
                    "cat": cat,
                    "score": 8.0 + (0.5 if cat == "surface" else 0.0),
                    "defects": '{"whitening": 0.1}' if cat == "corners" else "{}",
                })
            result.ok("INSERT condition_scores (4 categories) OK")

            # --- INSERT condition_calibration ---
            conn.execute(text("""
                INSERT INTO condition_calibration
                    (scan_id, grade_authority, actual_grade,
                     actual_subgrades, cert_number, predicted_grade)
                VALUES
                    (:scan_id, 'PSA', 9.0,
                     '{"centering": 9, "corners": 9, "edges": 9, "surface": 9}'::jsonb,
                     'TEST-12345', 8.5)
            """), {"scan_id": scan_id})
            result.ok("INSERT condition_calibration OK")

            # --- READ back and verify ---
            scan_row = conn.execute(text("""
                SELECT scan_id, card_id, overall_grade, tcg_condition,
                       confidence, model_version
                FROM condition_scans WHERE scan_id = :sid
            """), {"sid": scan_id}).fetchone()

            if scan_row and scan_row.overall_grade == 8.5:
                result.ok("READ condition_scans verified")
            else:
                result.fail(f"READ condition_scans mismatch: {scan_row}")

            score_count = conn.execute(text("""
                SELECT COUNT(*) FROM condition_scores WHERE scan_id = :sid
            """), {"sid": scan_id}).scalar()
            if score_count == 4:
                result.ok(f"READ condition_scores: {score_count} rows (expected 4)")
            else:
                result.fail(f"READ condition_scores: {score_count} rows (expected 4)")

            img_count = conn.execute(text("""
                SELECT COUNT(*) FROM condition_images WHERE scan_id = :sid
            """), {"sid": scan_id}).scalar()
            if img_count == 1:
                result.ok(f"READ condition_images: {img_count} row (expected 1)")
            else:
                result.fail(f"READ condition_images: {img_count} rows (expected 1)")

            calib_row = conn.execute(text("""
                SELECT actual_grade, cert_number
                FROM condition_calibration WHERE scan_id = :sid
            """), {"sid": scan_id}).fetchone()
            if calib_row and calib_row.actual_grade == 9.0:
                result.ok("READ condition_calibration verified")
            else:
                result.fail(f"READ condition_calibration mismatch: {calib_row}")

            # --- UPDATE ---
            conn.execute(text("""
                UPDATE condition_scans SET overall_grade = 9.0
                WHERE scan_id = :sid
            """), {"sid": scan_id})
            updated = conn.execute(text("""
                SELECT overall_grade FROM condition_scans WHERE scan_id = :sid
            """), {"sid": scan_id}).scalar()
            if updated == 9.0:
                result.ok("UPDATE condition_scans OK")
            else:
                result.fail(f"UPDATE condition_scans: expected 9.0, got {updated}")

            # --- Test UNIQUE constraint on condition_scores(scan_id, category) ---
            conn.execute(text("SAVEPOINT sp_unique"))
            try:
                conn.execute(text("""
                    INSERT INTO condition_scores (scan_id, category, score)
                    VALUES (:sid, 'centering', 7.0)
                """), {"sid": scan_id})
                conn.execute(text("ROLLBACK TO SAVEPOINT sp_unique"))
                result.fail("UNIQUE constraint on condition_scores NOT enforced")
            except Exception:
                conn.execute(text("ROLLBACK TO SAVEPOINT sp_unique"))
                result.ok("UNIQUE constraint on condition_scores enforced")

            # --- Test CASCADE DELETE ---
            conn.execute(text("""
                DELETE FROM condition_calibration WHERE scan_id = :sid
            """), {"sid": scan_id})
            conn.execute(text("""
                DELETE FROM condition_scans WHERE scan_id = :sid
            """), {"sid": scan_id})

            # Verify cascade deleted child rows
            remaining_scores = conn.execute(text("""
                SELECT COUNT(*) FROM condition_scores WHERE scan_id = :sid
            """), {"sid": scan_id}).scalar()
            remaining_images = conn.execute(text("""
                SELECT COUNT(*) FROM condition_images WHERE scan_id = :sid
            """), {"sid": scan_id}).scalar()

            if remaining_scores == 0:
                result.ok("CASCADE DELETE: condition_scores cleaned up")
            else:
                result.fail(
                    f"CASCADE DELETE: {remaining_scores} condition_scores remain"
                )

            if remaining_images == 0:
                result.ok("CASCADE DELETE: condition_images cleaned up")
            else:
                result.fail(
                    f"CASCADE DELETE: {remaining_images} condition_images remain"
                )

            # Rollback so we don't leave test data
            conn.rollback()
            result.ok("ROLLBACK: no test data left in DB")

        except Exception as e:
            conn.rollback()
            result.fail(f"CRUD test error: {e}")
            traceback.print_exc()


def check_constraints(result: TestResult):
    """Test CHECK constraints on enum-like columns."""
    print("\n[7] Testing CHECK constraints...")

    with engine.connect() as conn:
        card_id = conn.execute(
            text("SELECT card_id FROM dim_cards LIMIT 1")
        ).scalar()
        if not card_id:
            result.fail("No cards in dim_cards for constraint tests")
            return

        # --- tcg_condition must be one of NM/LP/MP/HP/DMG ---
        try:
            conn.execute(text("""
                INSERT INTO condition_scans (card_id, tcg_condition)
                VALUES (:cid, 'INVALID')
            """), {"cid": card_id})
            conn.rollback()
            result.fail("CHECK constraint on tcg_condition NOT enforced")
        except Exception:
            conn.rollback()
            result.ok("CHECK constraint on tcg_condition enforced")

        # --- angle_type must be valid ---
        try:
            # Need a scan_id first
            scan_id = conn.execute(text("""
                INSERT INTO condition_scans (card_id)
                VALUES (:cid)
                RETURNING scan_id
            """), {"cid": card_id}).scalar()

            conn.execute(text("""
                INSERT INTO condition_images (scan_id, angle_type, image_path)
                VALUES (:sid, 'INVALID_ANGLE', '/tmp/x.jpg')
            """), {"sid": scan_id})
            conn.rollback()
            result.fail("CHECK constraint on angle_type NOT enforced")
        except Exception:
            conn.rollback()
            result.ok("CHECK constraint on angle_type enforced")

        # --- category must be valid ---
        try:
            scan_id = conn.execute(text("""
                INSERT INTO condition_scans (card_id)
                VALUES (:cid)
                RETURNING scan_id
            """), {"cid": card_id}).scalar()

            conn.execute(text("""
                INSERT INTO condition_scores (scan_id, category, score)
                VALUES (:sid, 'INVALID_CAT', 5.0)
            """), {"sid": scan_id})
            conn.rollback()
            result.fail("CHECK constraint on category NOT enforced")
        except Exception:
            conn.rollback()
            result.ok("CHECK constraint on category enforced")

        # --- grade_authority must be valid ---
        try:
            scan_id = conn.execute(text("""
                INSERT INTO condition_scans (card_id)
                VALUES (:cid)
                RETURNING scan_id
            """), {"cid": card_id}).scalar()

            conn.execute(text("""
                INSERT INTO condition_calibration
                    (scan_id, grade_authority, actual_grade)
                VALUES (:sid, 'INVALID_AUTH', 9.0)
            """), {"sid": scan_id})
            conn.rollback()
            result.fail("CHECK constraint on grade_authority NOT enforced")
        except Exception:
            conn.rollback()
            result.ok("CHECK constraint on grade_authority enforced")


def check_fk_enforcement(result: TestResult):
    """Verify FK to dim_cards is enforced."""
    print("\n[8] Testing FK enforcement...")

    with engine.connect() as conn:
        try:
            conn.execute(text("""
                INSERT INTO condition_scans (card_id)
                VALUES ('NONEXISTENT-CARD-9999')
            """))
            conn.rollback()
            result.fail("FK to dim_cards NOT enforced")
        except Exception:
            conn.rollback()
            result.ok("FK to dim_cards enforced (rejects invalid card_id)")


def print_schema_summary(inspector):
    """Print a summary of the condition tables for reference."""
    print("\n" + "=" * 60)
    print("CONDITION SCHEMA SUMMARY")
    print("=" * 60)
    for tbl in CONDITION_TABLES:
        try:
            cols = inspector.get_columns(tbl)
            fks = inspector.get_foreign_keys(tbl)
            indexes = inspector.get_indexes(tbl)
            pk = inspector.get_pk_constraint(tbl)

            print(f"\n  {tbl}")
            print(f"  {'─' * len(tbl)}")
            print(f"  PK: {pk.get('constrained_columns', [])}")
            for c in cols:
                nullable = "" if c.get("nullable", True) else " NOT NULL"
                default = f" DEFAULT {c['default']}" if c.get("default") else ""
                print(f"    {c['name']:<20s} {str(c['type']):<30s}{nullable}{default}")
            if fks:
                print(f"  FKs: {', '.join(f['referred_table'] for f in fks)}")
            if indexes:
                print(f"  Indexes: {', '.join(i['name'] for i in indexes)}")
        except Exception as e:
            print(f"\n  {tbl}: ERROR - {e}")


def main():
    parser = argparse.ArgumentParser(description="Condition DB schema validator")
    parser.add_argument(
        "--migrate", action="store_true",
        help="Run migration before checks"
    )
    args = parser.parse_args()

    # Check DB connectivity first
    print("Connecting to database...")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("  Database connection OK")
    except Exception as e:
        print(f"  FATAL: Cannot connect to database: {e}")
        print("\n  Is PostgreSQL running?")
        print("  Try: sudo service postgresql start")
        sys.exit(1)

    # Optionally run migration
    if args.migrate:
        print("\nRunning migration...")
        from cardprice.db.migrate import run as run_migration
        run_migration()

    inspector = inspect(engine)
    result = TestResult()

    # Check all existing tables first
    existing_tables = set(inspector.get_table_names())
    missing = set(CONDITION_TABLES.keys()) - existing_tables
    if missing:
        print(f"\nMissing tables: {missing}")
        print("Run with --migrate to create them, or:")
        print("  python -c 'from cardprice.db.migrate import run; run()'")

        if not args.migrate:
            # Still run checks to report failures properly
            pass

    # Run all checks
    check_tables_exist(inspector, result)

    # Only run detailed checks if tables exist
    if not missing:
        check_columns(inspector, result)
        check_primary_keys(inspector, result)
        check_foreign_keys(inspector, result)
        check_indexes(inspector, result)
        check_crud(result)
        check_constraints(result)
        check_fk_enforcement(result)
        print_schema_summary(inspector)

    success = result.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
