"""Product orchestration accepts separate correction configuration."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from dolphin.workflows.config import CorrectionOptions, DisplacementWorkflow
from disp_s1 import main


@pytest.mark.parametrize('legacy', [False, True])
def test_geometry_reaches_packaging(tmp_path, monkeypatch, legacy):
    cfg=DisplacementWorkflow(cslc_file_list=[tmp_path/'20240101.tif'],work_directory=tmp_path)
    options=CorrectionOptions(geometry_files=[tmp_path/'static.h5'])
    if legacy:
        cfg=SimpleNamespace(**cfg.__dict__,correction_options=options)
    pair=tmp_path/'20240101_20240113.tif'
    outputs=SimpleNamespace(timeseries_paths=[pair],ionospheric_corrections=None,
                            stitched_cor_paths=[pair],conncomp_paths=[pair])
    runconfig=SimpleNamespace(
        product_path_group=SimpleNamespace(output_directory=tmp_path/'products',save_compressed_slc=False),
        dynamic_ancillary_file_group=SimpleNamespace(mask_file=None,algorithm_parameters_file=tmp_path/'algorithm.yaml'))
    monkeypatch.setattr(main,'read_reference_point',lambda _:None)
    monkeypatch.setattr(main,'_create_nodata_mask',lambda **kw:None)
    seen=[]
    def incidence(files):
        seen.append(files)
        return 31.,42.
    monkeypatch.setattr(main,'_get_near_far_incidence_angles',incidence)
    monkeypatch.setattr(main.AlgorithmParameters,'from_yaml',lambda _:SimpleNamespace(num_parallel_products=1))
    packaged=[]
    monkeypatch.setattr(main,'create_displacement_products',lambda *a,**kw:packaged.append(kw))
    metadata = main.product.ProductMetadata({}, lambda **kw: None, lambda **kw: None)
    main.create_products(outputs,cfg,runconfig,product_metadata=metadata,**({} if legacy else {'correction_options':options}))
    assert packaged[0]['product_metadata'] is metadata
    assert seen==[options.geometry_files]
    assert packaged[0]['near_far_incidence_angles']==(31.,42.)
    # Explicit empty geometry must override legacy geometry, not fall back to it.
    if legacy:
        main.create_products(outputs,cfg,runconfig,correction_options=CorrectionOptions())
        assert packaged[-1]['product_metadata'] is None
        assert len(seen)==1
        assert packaged[-1]['near_far_incidence_angles']==(30.,45.)


def test_missing_correction_configuration_fails_before_io():
    with pytest.raises(ValueError,match='Pass correction_options explicitly'):
        main.create_products(None,SimpleNamespace(),None)


def test_official_metadata_defaults(tmp_path):
    import inspect
    import h5py
    from disp_s1 import product
    algorithm = tmp_path / 'algorithm.yaml'
    algorithm.write_text('ps_options: {}\n')
    cfg = SimpleNamespace(
        dynamic_ancillary_file_group=SimpleNamespace(algorithm_parameters_file=algorithm),
        to_yaml=lambda stream: stream.write('official_config: true\n'))
    output = tmp_path / 'official.nc'
    workflow = DisplacementWorkflow(cslc_file_list=[tmp_path / "20240101.tif"], work_directory=tmp_path / "work")
    product._create_metadata_group(output, cfg, workflow)
    with h5py.File(output) as f:
        assert f['metadata/product_landing_page_doi'].asstr()[()] == 'https://doi.org/10.5067/SNWG/OPL3DISPS1-V1'
        assert f['metadata/product_specification_document_id'].asstr()[()] == 'JPL D-108772'
        assert f['metadata/pge_runconfig'].asstr()[()] == 'official_config: true\n'
        assert 'local_product_config' not in f['metadata']
    assert product.GLOBAL_ATTRS['institution'] == 'NASA JPL'
    assert inspect.signature(product.create_output_product).parameters['product_metadata'].default is None
