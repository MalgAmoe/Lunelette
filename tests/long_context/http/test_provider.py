import unittest
from unittest.mock import patch

from provider import Tokenizer, check_version


class Tests(unittest.TestCase):
    def test_version(self):
        check_version()
        with patch('provider.version', return_value='1.13.0'):
            with self.assertRaisesRegex(RuntimeError, '1.13.1'):
                check_version()

    def test_decode_policy(self):
        t = Tokenizer.__new__(Tokenizer)
        class Pieces:
            def id_to_byte_piece(self, token):
                if token == 2:
                    raise AssertionError('EOS is not a byte piece')
                return {10:b'hello', 11:b'\xe2', 12:b'\x82\xac', 13:b'\0', 14:b'\xff'}[token]
        t.tokenizer = Pieces()
        self.assertEqual(t.decode([10, 2]), dict(text='hello', incomplete_utf8=False))
        self.assertEqual(t.decode([11]), dict(text='', incomplete_utf8=True))
        with self.assertRaises(ValueError):
            t.decode([2])
        self.assertEqual(t.decode([12, 2]), dict(text='€', incomplete_utf8=False))
        for ids in ([2, 10], [13], [14]):
            t.reset_decode()
            with self.assertRaises(ValueError):
                t.decode(ids)


if __name__ == '__main__':
    unittest.main()
