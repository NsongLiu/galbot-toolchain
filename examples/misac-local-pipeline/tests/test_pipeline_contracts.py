"""Synthetic contracts; no robot, recording, checkpoint or LeRobot dependency."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


pipeline = module('pipeline_contract', 'run_all.py')
splitter = module('split_contract', 'scripts/prepare_misac_finetune.py')
config_loader = module('config_contract', 'galbot/deploy/config_loader.py')
converter = module('conversion_contract', 'convert_mcap_to_lerobot.py')


class PipelineContracts(unittest.TestCase):
    def test_episode_split_is_disjoint_complete_and_repeatable(self):
        episodes = [{'task': task, 'episode_index': index}
                    for task, values in [('synthetic_a', range(5)), ('synthetic_b', range(5, 11))]
                    for index in values]
        result = splitter._stratified_split(episodes, 0.2, 1000)
        self.assertEqual(result, splitter._stratified_split(episodes, 0.2, 1000))
        train, evaluation = set(result['train_episodes']), set(result['eval_episodes'])
        self.assertFalse(train & evaluation)
        self.assertEqual(train | evaluation, set(range(11)))
        for task in result['tasks'].values():
            self.assertTrue(task['train'])
            self.assertTrue(task['eval'])

    def test_single_episode_task_stays_in_training(self):
        split = splitter._stratified_split([{'task': 'synthetic', 'episode_index': 0}], 0.2, 1000)
        self.assertEqual(split['train_episodes'], [0])
        self.assertEqual(split['eval_episodes'], [])

    def test_destructive_operations_reject_root_and_external_paths(self):
        with self.assertRaises(RuntimeError):
            pipeline._safe_rmtree(ROOT)
        with tempfile.TemporaryDirectory() as raw:
            external = Path(raw)
            sentinel = external / 'keep.txt'
            sentinel.write_text('synthetic', encoding='utf-8')
            with self.assertRaises(RuntimeError):
                pipeline._safe_rmtree(external)
            self.assertEqual(sentinel.read_text(encoding='utf-8'), 'synthetic')
            config = pipeline._load_config()
            config['dataset_root'] = str(external)
            with self.assertRaises(RuntimeError):
                pipeline._resolve_config(config)

    def test_configuration_override_keeps_trace_defaults(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / 'deploy.yaml'
            path.write_text('mode: replay_debug\ndataset_root: ./synthetic_data\nimage_save:\n  enabled: true\n', encoding='utf-8')
            config = config_loader.load_config(path)
            self.assertEqual(config['dataset_root'], './synthetic_data')
            self.assertTrue(config['image_save']['enabled'])
            self.assertEqual(config['image_save']['interval'], 1)
            path.write_text('mode: invalid_mode\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                config_loader.load_config(path)

    def test_supported_action_and_camera_selection(self):
        self.assertEqual(converter.parse_action_groups('right_arm'), [('right_arm', 7)])
        self.assertEqual(converter.parse_camera_keys('right_arm'), ['right_arm'])
        with self.assertRaises(ValueError):
            converter.parse_action_groups('unknown_group')
        with self.assertRaises(ValueError):
            converter.parse_camera_keys('unknown_camera')


if __name__ == '__main__':
    unittest.main()
