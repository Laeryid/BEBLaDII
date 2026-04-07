import os
import shutil
import pytest
import torch
import json
from beb_la_dii.model.base import BEComponent
from beb_la_dii.model.component_registry import ComponentRegistry
from beb_la_dii.utils.experiment_manager import ExperimentManager

@pytest.fixture
def temp_storage():
    storage = "storage/pytest_temp"
    os.makedirs(storage, exist_ok=True)
    yield storage
    if os.path.exists(storage):
        shutil.rmtree(storage)

def test_be_component_metadata():
    config = {"dim": 512, "layers": 12}
    comp = BEComponent(component_id="test_comp", version="v1.2", config=config)
    
    meta = comp.get_metadata()
    assert meta["component_id"] == "test_comp"
    assert meta["version"] == "v1.2"
    assert meta["config"] == config
    assert meta["class_name"] == "BEComponent"

def test_component_registry_save_load(temp_storage):
    registry = ComponentRegistry(storage_root=temp_storage)
    
    # Mock component inheriting from BEComponent but being an nn.Module
    class MockComp(BEComponent, torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.param = torch.nn.Parameter(torch.ones(1))

    comp = MockComp(component_id="mock", version="v1.0", config={"a": 1})
    
    # Save
    registry.save_component(comp, "test_type")
    
    # Load
    loaded = registry.load_component(MockComp, "test_type", "mock", "v1.0")
    
    assert loaded.component_id == "mock"
    assert loaded.version == "v1.0"
    assert loaded.config == {"a": 1}
    assert torch.equal(loaded.param, comp.param)

def test_registry_version_isolation(temp_storage):
    registry = ComponentRegistry(storage_root=temp_storage)
    class MockComp(BEComponent, torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.param = torch.nn.Parameter(torch.ones(1))

    comp_v1 = MockComp(component_id="iso", version="v1.0")
    comp_v2 = MockComp(component_id="iso", version="v2.0")
    comp_v2.param.data.fill_(2.0)

    registry.save_component(comp_v1, "type")
    registry.save_component(comp_v2, "type")

    versions = registry.list_versions("type", "iso")
    assert "v1.0" in versions
    assert "v2.0" in versions

    loaded_v1 = registry.load_component(MockComp, "type", "iso", "v1.0")
    assert loaded_v1.param.item() == 1.0

def test_experiment_manager_snapshots(temp_storage):
    registry_path = os.path.join(temp_storage, "components")
    exp_path = os.path.join(temp_storage, "experiments")
    
    registry = ComponentRegistry(storage_root=registry_path)
    manager = ExperimentManager(experiment_root=exp_path, registry=registry)
    
    config = {"task": "test", "lr": 1e-4}
    exp_id, full_path = manager.create_experiment("test_exp", config)
    
    assert os.path.exists(os.path.join(full_path, "config.json"))
    
    # Snapshot
    results = {"loss": 0.5}
    comp = BEComponent(component_id="c1", version="v1")
    
    snap_path = manager.save_snapshot(full_path, {"comp_1": comp}, results)
    assert os.path.exists(snap_path)
    
    with open(snap_path, "r") as f:
        snap_data = json.load(f)
    
    assert snap_data["results"]["loss"] == 0.5
    assert snap_data["composition"]["comp_1"]["id"] == "c1"
