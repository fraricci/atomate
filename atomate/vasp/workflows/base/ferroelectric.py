"""
This module defines the ferroelectric workflow
"""

from fireworks import Firework, Workflow

from atomate.common.powerups import add_tags
from atomate.utils.utils import get_a_unique_id, get_logger
from atomate.vasp.firetasks.parse_outputs import PolarizationToDb
from atomate.vasp.fireworks.core import HSEBSFW, OptimizeFW
from atomate.vasp.fireworks.polarization import LcalcpolFW

__author__ = "Tess Smidt"
__email__ = "tsmidt@berkeley.edu"

logger = get_logger(__name__)


def get_wf_ferroelectric(
    polar_structure,
    nonpolar_structure,
    vasp_cmd="vasp",
    db_file=None,
    vasp_input_set_polar="MPStaticSet",
    vasp_input_set_nonpolar="MPStaticSet",
    vasp_input_set_params_polar=None,
    vasp_input_set_params_nonpolar=None,
    relax=False,
    vasp_relax_input_set_polar=None,
    vasp_relax_input_set_nonpolar=None,
    nimages=9,
    hse=False,
    add_analysis_task=False,
    wfid=None,
    tags=None,
    exit_fw=True,
    defuse_wf=True,
    
):
    """
    Returns a workflow to calculate the spontaneous polarization of polar_structure using
    a nonpolar reference phase structure and linear interpolations between the polar and
    nonpolar structure.

    The nonpolar and polar structures must be in the same space group setting and atoms ordered
    such that a linear interpolation can be performed to create intermediate structures along
    the distortion.

    For example, to calculate the polarization of orthorhombic BaTiO3 (space group 38) using
    the cubic structure (space group 221) as the nonpolar reference phase, we must transform
    the cubic to the orthorhombic setting. This can be accomplished using Bilbao Crystallographic
    Server's Structure Relations tool. (http://www.cryst.ehu.es/cryst/rel.html)

    Args:
        polar_structure (Structure): polar structure of candidate ferroelectric
        nonpolar_structure (Structure): nonpolar reference structure in polar setting
        vasp_input_set_polar (DictVaspInputSet): VASP polar input set. Defaults to MPStaticSet.
        vasp_input_set_nonpolar (DictVaspInputSet): VASP nonpolar input set. Defaults to MPStaticSet.
        vasp_relax_input_set_polar (DictVaspInputSet): VASP polar input set. Defaults to MPRelaxSet.
        vasp_relax_input_set_nonpolar (DictVaspInputSet): VASP nonpolar input set. Defaults to MPRelaxSet.
        relax (bool or list of bools): if single bool, relax or not both polar and nonpolar; if list,
                each bool controls individual relaxation for polar and nonpolar, respectively.
        vasp_cmd (str): command to run
        db_file (str): path to file containing the database credentials.
        nimages: Number of interpolations calculated from polar to nonpolar structures, including the nonpolar.
            For example, nimages = 9 will calculate 8 interpolated structures. 8 interpolations + nonpolar = 9.
        add_analysis_task: Analyze polarization and energy trends as part of workflow. Default False.
        wfid (string): Unique workflow id starting with "wfid_". If None this is automatically generated (recommended).
        tags (list of strings): Additional tags to add such as identifiers for structures.
        exit_fw (bool), exit the lcalcpol fws in case the band gap is below a certain threshold. Default True.
        defuse_wf (bool), defuse entire wf in case the band gap of any of the structures is below a certain threshold. Default True.

    Returns:

    """
    wf = []

    if wfid is None:
        wfid = "wfid_" + get_a_unique_id()
    if tags is None:
        tags = []

    if isinstance(relax,bool):
        relax = (relax,relax)
        
    if relax[0]:
        polar_relax = OptimizeFW(
            structure=polar_structure,
            name="_polar_relaxation",
            vasp_cmd=vasp_cmd,
            db_file=db_file,
            vasp_input_set=vasp_relax_input_set_polar,
        )
        wf.append(polar_relax)
        parents_polar = polar_relax
    else:
        parents_polar = None
        
    if relax[1]:
        nonpolar_relax = OptimizeFW(
            structure=nonpolar_structure,
            name="_nonpolar_relaxation",
            vasp_cmd=vasp_cmd,
            db_file=db_file,
            vasp_input_set=vasp_relax_input_set_nonpolar,
        )
        wf.append(nonpolar_relax)
        parents_nonpolar = nonpolar_relax
    else:
        parents_nonpolar = None

    # Run polarization calculation on polar structure.
    # Defuse workflow if polar structure is metallic.
    polar = LcalcpolFW(
        structure=polar_structure,
        name="_polar_polarization",
        static_name="_polar_static",
        parents=parents_polar,
        vasp_cmd=vasp_cmd,
        db_file=db_file,
        vasp_input_set=vasp_input_set_polar,
        vasp_input_set_params=vasp_input_set_params_polar,
        exit_fw=exit_fw,
        defuse_wf=defuse_wf,
    )

    # Run polarization calculation on nonpolar structure.
    # Defuse workflow if nonpolar structure is metallic.
    nonpolar = LcalcpolFW(
        structure=nonpolar_structure,
        name="_nonpolar_polarization",
        static_name="_nonpolar_static",
        parents=parents_nonpolar,
        vasp_cmd=vasp_cmd,
        db_file=db_file,
        vasp_input_set=vasp_input_set_nonpolar,
        vasp_input_set_params=vasp_input_set_params_nonpolar,
        exit_fw=exit_fw,
        defuse_wf=defuse_wf,
    )

    # Interpolation polarization
    interpolation = []
    # Interpolations start from one increment after polar and end prior to nonpolar.
    # The Structure.interpolate method adds an additional image for the nonpolar endpoint.
    # Defuse children fireworks if metallic.
    for i in range(1, nimages):
        # nonpolar_structure is being used as a dummy structure.
        # The structure will be replaced by the interpolated structure generated by
        # StaticInterpolatedFW.
        # Defuse workflow if interpolated structure is metallic.
        interpolation.append(
            LcalcpolFW(
                structure=polar_structure,
                name=f"_interpolation_{i}_polarization",
                static_name=f"_interpolation_{i}_static",
                vasp_cmd=vasp_cmd,
                db_file=db_file,
                vasp_input_set=vasp_input_set_polar,
                vasp_input_set_params=vasp_input_set_params_polar,
                interpolate=True,
                start="_polar_static",
                end="_nonpolar_static",
                nimages=nimages,
                this_image=i,
                exit_fw=exit_fw,
                defuse_wf=defuse_wf,
                parents=[polar, nonpolar],
            )
        )

    wf.append(polar)
    wf.append(nonpolar)
    wf += interpolation

    # Add FireTask that uses Polarization object to store spontaneous polarization information
    if add_analysis_task:
        fw_analysis = Firework(
            PolarizationToDb(db_file=db_file),
            parents=interpolation,
            name="_polarization_post_processing",
        )
        wf.append(fw_analysis)

    # Run HSE band gap calculation
    if hse:
        # Run HSE calculation at band gap for polar calculation if polar structure is not metallic
        hse = HSEBSFW(
            structure=polar_structure,
            parents=polar,
            name="_polar_hse_gap",
            vasp_cmd=vasp_cmd,
            db_file=db_file,
            calc_loc="_polar_polarization",
        )
        wf.append(hse)

    # Create Workflow task and add tags to workflow
    workflow = Workflow(wf)
    workflow = add_tags(workflow, [wfid] + tags)

    return workflow
