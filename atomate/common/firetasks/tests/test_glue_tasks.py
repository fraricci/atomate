import os
import unittest

from fireworks import Firework, Workflow
from fireworks.core.rocket_launcher import rapidfire
from monty.tempfile import ScratchDir

from atomate.common.firetasks.glue_tasks import (
    CopyFilesFromCalcLoc,
    CreateFolder,
    DeleteFiles,
    DeleteFilesPrevFolder,
    PassCalcLocs,
    get_calc_loc,
)
from atomate.utils.testing import AtomateTest
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs

__author__ = "Anubhav Jain <ajain@lbl.gov>"

module_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(module_dir, "..", "..", "test_files")


class TestPassCalcLocs(AtomateTest):
    def test_passcalclocs(self):
        fw1 = Firework([PassCalcLocs(name="fw1")], name="fw1")
        fw2 = Firework([PassCalcLocs(name="fw2")], name="fw2", parents=fw1)
        fw3 = Firework([PassCalcLocs(name="fw3")], name="fw3", parents=fw2)

        wf = Workflow([fw1, fw2, fw3])
        self.lp.add_wf(wf)
        rapidfire(self.lp)

        fw2 = self.lp.get_fw_by_id(self.lp.get_fw_ids({"name": "fw2"})[0])
        fw3 = self.lp.get_fw_by_id(self.lp.get_fw_ids({"name": "fw3"})[0])

        self.assertEqual(len(fw2.spec["calc_locs"]), 1)
        self.assertEqual(len(fw3.spec["calc_locs"]), 2)
        self.assertEqual(fw3.spec["calc_locs"][0]["name"], "fw1")
        self.assertEqual(fw3.spec["calc_locs"][1]["name"], "fw2")
        self.assertNotEqual(
            fw3.spec["calc_locs"][0]["path"], fw3.spec["calc_locs"][1]["path"]
        )

        calc_locs = fw3.spec["calc_locs"]
        self.assertEqual(get_calc_loc("fw1", calc_locs), calc_locs[0])
        self.assertEqual(get_calc_loc("fw2", calc_locs), calc_locs[1])
        self.assertEqual(get_calc_loc(True, calc_locs), calc_locs[1])


class TestDeleteFiles(AtomateTest):
    def test_cleanupfiles(self):
        fw1 = Firework(
            [
                CreateFolder(folder_name="to_remove.relax0"),
                CreateFolder(folder_name="to_remove.relax1"),
                CreateFolder(folder_name="dont_remove.relax0"),
                CreateFolder(folder_name="shouldnt_touch"),
                DeleteFiles(files=["to_remove*", "dont_remove"]),
                PassCalcLocs(name="fw1"),
            ],
            name="fw1",
        )
        fw2 = Firework([PassCalcLocs(name="fw2")], name="fw2", parents=fw1)

        wf = Workflow([fw1, fw2])
        self.lp.add_wf(wf)
        rapidfire(self.lp)

        fw2 = self.lp.get_fw_by_id(self.lp.get_fw_ids({"name": "fw2"})[0])
        calc_locs = fw2.spec["calc_locs"]

        self.assertTrue(
            os.path.exists(
                os.path.join(
                    get_calc_loc("fw1", calc_locs)["path"], "dont_remove.relax0"
                )
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(get_calc_loc("fw1", calc_locs)["path"], "shouldnt_touch")
            )
        )
        self.assertFalse(
            os.path.exists(
                os.path.join(get_calc_loc("fw1", calc_locs)["path"], "to_remove.relax0")
            )
        )
        self.assertFalse(
            os.path.exists(
                os.path.join(get_calc_loc("fw1", calc_locs)["path"], "to_remove.relax1")
            )
        )


class TestDeleteFilesPrevFolder(AtomateTest):
    def test_cleanupfiles(self):
        fw1 = Firework(
            [
                CreateFolder(folder_name="to_remove.relax0"),
                CreateFolder(folder_name="to_remove.relax1"),
                CreateFolder(folder_name="dont_remove.relax0"),
                CreateFolder(folder_name="shouldnt_touch"),
                DeleteFilesPrevFolder(files=["to_remove*", "dont_remove"]),
                PassCalcLocs(name="fw1"),
            ],
            name="fw1",
        )
        fw2 = Firework([PassCalcLocs(name="fw2")], name="fw2", parents=fw1)

        wf = Workflow([fw1, fw2])
        self.lp.add_wf(wf)
        rapidfire(self.lp)

        fw2 = self.lp.get_fw_by_id(self.lp.get_fw_ids({"name": "fw2"})[0])
        calc_locs = fw2.spec["calc_locs"]

        self.assertTrue(
            os.path.exists(
                os.path.join(
                    get_calc_loc("fw1", calc_locs)["path"], "dont_remove.relax0"
                )
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(get_calc_loc("fw1", calc_locs)["path"], "shouldnt_touch")
            )
        )
        self.assertFalse(
            os.path.exists(
                os.path.join(get_calc_loc("fw1", calc_locs)["path"], "to_remove.relax0")
            )
        )
        self.assertFalse(
            os.path.exists(
                os.path.join(get_calc_loc("fw1", calc_locs)["path"], "to_remove.relax1")
            )
        )

    def test_cleanupfiles_calc_loc(self):
        # will test deleting files from previous folder
        fw1 = Firework(
            [
                CreateFolder(folder_name="to_remove.relax0"),
                CreateFolder(folder_name="to_remove.relax1"),
                CreateFolder(folder_name="dont_remove.relax0"),
                CreateFolder(folder_name="shouldnt_touch"),
                PassCalcLocs(name="fw1"),
            ],
            name="fw1",
        )
        fw2 = Firework(
            [
                DeleteFilesPrevFolder(
                    files=["to_remove*", "dont_remove"], calc_loc=True
                ),
                PassCalcLocs(name="fw2"),
            ],
            name="fw2",
            parents=fw1,
        )

        wf = Workflow([fw1, fw2])
        self.lp.add_wf(wf)
        rapidfire(self.lp)

        fw2 = self.lp.get_fw_by_id(self.lp.get_fw_ids({"name": "fw2"})[0])
        calc_locs = fw2.spec["calc_locs"]

        self.assertTrue(
            os.path.exists(
                os.path.join(
                    get_calc_loc("fw1", calc_locs)["path"], "dont_remove.relax0"
                )
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(get_calc_loc("fw1", calc_locs)["path"], "shouldnt_touch")
            )
        )
        self.assertFalse(
            os.path.exists(
                os.path.join(get_calc_loc("fw1", calc_locs)["path"], "to_remove.relax0")
            )
        )
        self.assertFalse(
            os.path.exists(
                os.path.join(get_calc_loc("fw1", calc_locs)["path"], "to_remove.relax1")
            )
        )

    def test_cleanupfiles_calc_dir(self):
        # will test deleting from some folder specified by calc_dir
        with ScratchDir(".", copy_from_current_on_enter=True):
            current_path = os.getcwd()
            os.mkdir("to_remove.relax0")
            os.mkdir("to_remove.relax1")
            os.mkdir("dont_remove.relax0")
            os.mkdir("shouldnt_touch")

            fw1 = Firework(
                [
                    DeleteFilesPrevFolder(
                        files=["to_remove*", "dont_remove"], calc_dir=current_path
                    ),
                    PassCalcLocs(name="fw1"),
                ],
                name="fw1",
            )
            fw2 = Firework([PassCalcLocs(name="fw2")], name="fw2", parents=fw1)

            wf = Workflow([fw1, fw2])
            self.lp.add_wf(wf)
            rapidfire(self.lp)

            self.assertTrue(
                os.path.exists(os.path.join(current_path, "dont_remove.relax0"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(current_path, "shouldnt_touch"))
            )
            self.assertFalse(
                os.path.exists(os.path.join(current_path, "to_remove.relax0"))
            )
            self.assertFalse(
                os.path.exists(os.path.join(current_path, "to_remove.relax1"))
            )


class TestCreateFolder(AtomateTest):
    def test_createfolder(self):
        folder_name = "test_folder"
        fw1 = Firework(
            [
                CreateFolder(folder_name=folder_name, change_dir=False),
                PassCalcLocs(name="fw1"),
            ],
            name="fw3",
        )
        fw2 = Firework([PassCalcLocs(name="fw2")], name="fw2", parents=fw1)
        wf = Workflow([fw1, fw2])
        self.lp.add_wf(wf)
        rapidfire(self.lp)

        fw2 = self.lp.get_fw_by_id(self.lp.get_fw_ids({"name": "fw2"})[0])
        calc_locs = fw2.spec["calc_locs"]

        self.assertTrue(
            os.path.exists(get_calc_loc("fw1", calc_locs)["path"] + "/" + folder_name)
        )


class TestCopyFilesFromCalcLoc(AtomateTest):
    @classmethod
    def setUpClass(cls):
        cls.plain_outdir = os.path.join(
            module_dir,
            "..",
            "..",
            "..",
            "vasp",
            "test_files",
            "Si_structure_optimization_plain",
            "outputs",
        )
        cls.relax2_outdir = os.path.join(
            module_dir,
            "..",
            "..",
            "..",
            "vasp",
            "test_files",
            "Si_structure_optimization_relax2",
            "outputs",
        )

    def test_copyfilesfromcalcloc(self):
        fw1 = Firework(
            [CopyVaspOutputs(calc_dir=self.plain_outdir), PassCalcLocs(name="fw1")],
            name="fw1",
        )

        fw2 = Firework(
            [CopyVaspOutputs(calc_dir=self.relax2_outdir), PassCalcLocs(name="fw2")],
            name="fw2",
        )

        fw3 = Firework(
            [
                CopyFilesFromCalcLoc(
                    calc_loc="fw1",
                    filenames=["POSCAR"],
                    name_prepend="",
                    name_append="_0",
                ),
                CopyFilesFromCalcLoc(
                    calc_loc="fw2",
                    filenames=["POSCAR"],
                    name_prepend="",
                    name_append="_1",
                ),
                PassCalcLocs(name="fw3"),
            ],
            name="fw3",
            parents=[fw1, fw2],
        )
        fw4 = Firework([PassCalcLocs(name="fw4")], name="fw4", parents=fw3)

        wf = Workflow([fw1, fw2, fw3, fw4])
        self.lp.add_wf(wf)
        rapidfire(self.lp)

        fw4 = self.lp.get_fw_by_id(self.lp.get_fw_ids({"name": "fw4"})[0])

        calc_locs = fw4.spec["calc_locs"]
        self.assertTrue(
            os.path.exists(get_calc_loc("fw3", calc_locs)["path"] + "/POSCAR_0")
        )
        self.assertTrue(
            os.path.exists(get_calc_loc("fw3", calc_locs)["path"] + "/POSCAR_1")
        )


if __name__ == "__main__":
    unittest.main()
