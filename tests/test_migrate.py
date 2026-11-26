from pathlib import Path

from db.migrate import MIGRATIONS_DIR, list_migrations, pending_migrations


def test_list_migrations_is_sorted_and_sql_only(tmp_path: Path) -> None:
    for name in ("002_b.sql", "001_a.sql", "notes.md"):
        (tmp_path / name).write_text("")

    assert [p.name for p in list_migrations(tmp_path)] == ["001_a.sql", "002_b.sql"]


def test_pending_migrations_skips_applied(tmp_path: Path) -> None:
    migrations = [tmp_path / "001_a.sql", tmp_path / "002_b.sql"]

    assert pending_migrations(migrations, applied={"001_a.sql"}) == [tmp_path / "002_b.sql"]
    assert pending_migrations(migrations, applied={"001_a.sql", "002_b.sql"}) == []


def test_real_migrations_have_unique_ordered_prefixes() -> None:
    prefixes = [path.name.split("_", 1)[0] for path in list_migrations(MIGRATIONS_DIR)]

    assert prefixes, "no migrations found"
    assert all(prefix.isdigit() for prefix in prefixes)
    assert len(prefixes) == len(set(prefixes))
