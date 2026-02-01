#!/usr/bin/env python3
"""
shard_io.py - Shared module for reading and writing chess training data shards.

This module provides unified I/O for binary shard files used by:
- prepare_data.py (writing)
- nn_train.py (reading for training)
- nn_tests.py (reading for testing)
- verify_data.py (reading for verification)

Binary shard format:
    DNN Normal record:
        [score:int16][num_features:uint8][features:uint16[]]

    DNN Diagnostic record (marker=0xFF):
        [marker:uint8=0xFF][score:int16][stm:uint8][num_features:uint8][features:uint16[]]
        [fen_length:uint8][fen_bytes:char[]]

    NNUE Normal record:
        [score:int16][stm:uint8][num_white:uint8][white:uint16[]]
        [num_black:uint8][black:uint16[]]

    NNUE Diagnostic record (marker=0xFF):
        [marker:uint8=0xFF][score:int16][stm:uint8][num_white:uint8][white:uint16[]]
        [num_black:uint8][black:uint16[]][fen_length:uint8][fen_bytes:char[]]

Feature encoding:
    - Piece order (shared by DNN and NNUE): P=0, N=1, B=2, R=3, Q=4, K=5 (piece_type - 1)
    - DNN: 768 features = 12 planes × 64 squares, feature_idx = piece_idx * 64 + square
    - NNUE: 40,960 features = king_sq * 640 + piece_sq * 10 + (type_idx + color_idx * 5)
"""

import io
import struct
import glob
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterator
import zstandard as zstd

# Constants
DIAGNOSTIC_MARKER = 0xFF
DIAGNOSTIC_INTERVAL = 1000


class ShardWriter:
    """
    Writes positions to compressed binary shard files.

    Diagnostic records (with FEN) are written every DIAGNOSTIC_INTERVAL positions.
    """

    def __init__(self, output_dir: str, prefix: str, nn_type: str, positions_per_shard: int):
        self.output_dir = Path(output_dir) / nn_type.lower()
        self.prefix = prefix
        self.nn_type = nn_type.upper()
        self.positions_per_shard = positions_per_shard

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Current shard state
        self.current_shard_num = 0
        self.positions_in_current_shard = 0
        self.buffer = io.BytesIO()
        self.total_positions = 0

        # Compression context (reusable)
        self.cctx = zstd.ZstdCompressor(level=3)

    def _get_shard_path(self, shard_num: int) -> Path:
        return self.output_dir / f"{self.prefix}_{shard_num:04d}.bin.zst"

    def _flush_shard(self):
        """Write current buffer to a shard file."""
        if self.positions_in_current_shard == 0:
            return

        self.current_shard_num += 1
        shard_path = self._get_shard_path(self.current_shard_num)

        # Compress and write
        compressed = self.cctx.compress(self.buffer.getvalue())
        with open(shard_path, 'wb') as f:
            f.write(compressed)

        # Reset buffer
        self.buffer = io.BytesIO()
        self.positions_in_current_shard = 0

    def add_position(self, position: Dict[str, Any]):
        """Add a position to the current shard."""
        score_cp = position['score_cp']
        is_diagnostic = (self.total_positions % DIAGNOSTIC_INTERVAL == 0)

        if self.nn_type == "DNN":
            features = position['dnn_features']

            if is_diagnostic:
                # Diagnostic record with marker, STM, and FEN
                fen = position['fen']
                stm = position['stm']
                fen_bytes = fen.encode('utf-8')

                self.buffer.write(struct.pack('<B', DIAGNOSTIC_MARKER))
                self.buffer.write(struct.pack('<h', score_cp))
                self.buffer.write(struct.pack('<B', stm))
                self.buffer.write(struct.pack('<B', len(features)))
                for f in features:
                    self.buffer.write(struct.pack('<H', f))
                self.buffer.write(struct.pack('<B', len(fen_bytes)))
                self.buffer.write(fen_bytes)
            else:
                # Normal record
                self.buffer.write(struct.pack('<h', score_cp))
                self.buffer.write(struct.pack('<B', len(features)))
                for f in features:
                    self.buffer.write(struct.pack('<H', f))
        else:  # NNUE
            white_feat = position['nnue_white']
            black_feat = position['nnue_black']
            stm = position['stm']

            if is_diagnostic:
                # Diagnostic record with marker and FEN
                fen = position['fen']
                fen_bytes = fen.encode('utf-8')

                self.buffer.write(struct.pack('<B', DIAGNOSTIC_MARKER))
                self.buffer.write(struct.pack('<h', score_cp))
                self.buffer.write(struct.pack('<B', stm))
                self.buffer.write(struct.pack('<B', len(white_feat)))
                for f in white_feat:
                    self.buffer.write(struct.pack('<H', f))
                self.buffer.write(struct.pack('<B', len(black_feat)))
                for f in black_feat:
                    self.buffer.write(struct.pack('<H', f))
                self.buffer.write(struct.pack('<B', len(fen_bytes)))
                self.buffer.write(fen_bytes)
            else:
                # Normal record
                self.buffer.write(struct.pack('<h', score_cp))
                self.buffer.write(struct.pack('<B', stm))
                self.buffer.write(struct.pack('<B', len(white_feat)))
                for f in white_feat:
                    self.buffer.write(struct.pack('<H', f))
                self.buffer.write(struct.pack('<B', len(black_feat)))
                for f in black_feat:
                    self.buffer.write(struct.pack('<H', f))

        self.positions_in_current_shard += 1
        self.total_positions += 1

        # Check if shard is full
        if self.positions_in_current_shard >= self.positions_per_shard:
            self._flush_shard()

    def finalize(self):
        """Flush any remaining positions to a final shard."""
        if self.positions_in_current_shard > 0:
            self._flush_shard()

    def get_stats(self) -> Dict[str, int]:
        return {
            'total_positions': self.total_positions,
            'num_shards': self.current_shard_num
        }


class ShardReader:
    """
    Reads positions from compressed binary shard files.

    Handles both normal and diagnostic records transparently.
    """

    def __init__(self, nn_type: str):
        self.nn_type = nn_type.upper()
        self.dctx = zstd.ZstdDecompressor()

    def _decompress_shard(self, shard_path: str) -> io.BytesIO:
        """Decompress a shard file and return a BytesIO buffer."""
        with open(shard_path, 'rb') as f:
            reader = self.dctx.stream_reader(f)
            data = reader.read()
            reader.close()
        return io.BytesIO(data)

    def read_all_positions(self, shard_path: str, include_fen: bool = False,  skip_diagnostic = True) -> \
            List[Dict[str, Any]]:
        """
        Read all positions from a shard file.

        Args:
            shard_path: Path to the shard file
            include_fen: If True, include FEN in diagnostic records (otherwise skip it)

        Returns:
            List of position dicts
            :param include_fen:
            :param shard_path:
            :param skip_diagnostic:
        """
        buf = self._decompress_shard(shard_path)
        positions = []

        while True:
            pos = self._read_one_position(buf, include_fen)
            if pos is None:
                break

            if skip_diagnostic and pos.get('is_diagnostic', True): # diagnostic records are reserved for testing
                continue

            positions.append(pos)

        return positions

    def read_diagnostic_records(self, shard_path: str, max_records: int = 10) -> List[Dict[str, Any]]:
        """
        Read only diagnostic records from a shard file.

        Args:
            shard_path: Path to the shard file
            max_records: Maximum number of diagnostic records to return

        Returns:
            List of diagnostic position dicts (with FEN)
        """
        buf = self._decompress_shard(shard_path)
        records = []
        position_idx = 0

        while len(records) < max_records:
            pos = self._read_one_position(buf, include_fen=True)
            if pos is None:
                break

            if pos.get('is_diagnostic', False):
                pos['position_idx'] = position_idx
                records.append(pos)

            position_idx += 1

        return records

    def iter_positions(self, shard_path: str, include_fen: bool = False, skip_diagnostic = True) -> \
            Iterator[Dict[str, Any]]:
        """
        Iterate over positions in a shard file (memory efficient).

        Args:
            shard_path: Path to the shard file
            include_fen: If True, include FEN in diagnostic records

        Yields:
            Position dicts one at a time
            :param include_fen:
            :param shard_path:
            :param skip_diagnostic:
        """
        buf = self._decompress_shard(shard_path)

        while True:
            pos = self._read_one_position(buf, include_fen)
            if pos is None:
                break

            if skip_diagnostic and pos.get('is_diagnostic', True): # diagnostic records are reserved for testing
                continue

            yield pos

    def _read_one_position(self, buf: io.BytesIO, include_fen: bool = False) -> Optional[Dict[str, Any]]:
        """Read a single position from the buffer.

        Diagnostic records are identified by:
        1. First byte is 0xFF (marker)
        2. For DNN: The byte after score+stm (num_features) must be <= 32
        3. The record must parse correctly

        For old format shards (without diagnostic records), 0xFF might appear
        as part of a score value, so we need to validate carefully.
        """
        first_byte = buf.read(1)
        if len(first_byte) < 1:
            return None

        first_val = struct.unpack('<B', first_byte)[0]

        if first_val == DIAGNOSTIC_MARKER:
            # Might be a diagnostic record - try to validate
            # Save position in case we need to revert
            saved_pos = buf.tell()

            try:
                result = self._try_read_diagnostic_record(buf, include_fen)
                if result is not None:
                    return result
            except Exception:
                pass

            # Not a valid diagnostic record, revert and read as normal
            # The 0xFF was actually part of a score
            buf.seek(saved_pos - 1)  # Go back to re-read first_byte as part of score
            first_byte = buf.read(1)
            return self._read_normal_record(buf, first_byte)
        else:
            return self._read_normal_record(buf, first_byte)

    def _try_read_diagnostic_record(self, buf: io.BytesIO, include_fen: bool = False) -> Optional[Dict[str, Any]]:
        """Try to read a diagnostic record. Returns None if validation fails.

        IMPORTANT: Validation must be strict enough to reject random data that
        happens to start with 0xFF. False positives corrupt parsing.
        """
        score_cp = struct.unpack('<h', buf.read(2))[0]
        stm = struct.unpack('<B', buf.read(1))[0]

        # Validate STM (must be 0 or 1)
        if stm not in (0, 1):
            return None

        if self.nn_type == "DNN":
            num_features = struct.unpack('<B', buf.read(1))[0]

            # Validate: chess positions have at most 32 pieces
            if num_features > 32 or num_features < 2:  # At minimum: 2 kings
                return None

            features = []
            for _ in range(num_features):
                feat = struct.unpack('<H', buf.read(2))[0]
                # Validate: DNN features must be < 768
                if feat >= 768:
                    return None
                features.append(feat)

            fen_length = struct.unpack('<B', buf.read(1))[0]

            # Validate: FEN strings are typically 20-90 characters
            if fen_length < 15 or fen_length > 100:
                return None

            fen_bytes = buf.read(fen_length)
            if len(fen_bytes) != fen_length:
                return None

            # Try to decode as UTF-8 and validate it looks like a FEN
            try:
                fen = fen_bytes.decode('utf-8')
                # Basic FEN validation: should contain '/' and space
                if '/' not in fen or ' ' not in fen:
                    return None
                # Additional FEN validation: should have 7 slashes (8 ranks)
                if fen.count('/') != 7:
                    return None
            except UnicodeDecodeError:
                return None

            result = {
                'score_cp': score_cp,
                'stm': stm,
                'features': features,
                'is_diagnostic': True
            }

            if include_fen:
                result['fen'] = fen

            return result
        else:  # NNUE
            num_white = struct.unpack('<B', buf.read(1))[0]

            # Validate: NNUE features are all non-king pieces from one perspective
            # Maximum is 30 pieces (15 per side × 2 sides, excluding both kings)
            if num_white > 30:
                return None

            white_features = []
            for _ in range(num_white):
                feat = struct.unpack('<H', buf.read(2))[0]
                # Validate: NNUE features must be < 40960
                if feat >= 40960:
                    return None
                white_features.append(feat)

            num_black = struct.unpack('<B', buf.read(1))[0]
            if num_black > 30:
                return None

            black_features = []
            for _ in range(num_black):
                feat = struct.unpack('<H', buf.read(2))[0]
                if feat >= 40960:
                    return None
                black_features.append(feat)

            # Note: num_white and num_black should be equal (same pieces, different perspectives)
            # but we don't enforce this strictly to handle edge cases

            fen_length = struct.unpack('<B', buf.read(1))[0]
            if fen_length < 15 or fen_length > 100:
                return None

            fen_bytes = buf.read(fen_length)
            if len(fen_bytes) != fen_length:
                return None

            try:
                fen = fen_bytes.decode('utf-8')
                if '/' not in fen or ' ' not in fen:
                    return None
                # Additional FEN validation: should have 7 slashes (8 ranks)
                if fen.count('/') != 7:
                    return None
            except UnicodeDecodeError:
                return None

            result = {
                'score_cp': score_cp,
                'stm': stm,
                'white_features': white_features,
                'black_features': black_features,
                'is_diagnostic': True
            }

            if include_fen:
                result['fen'] = fen

            return result

    def _read_normal_record(self, buf: io.BytesIO, first_byte: bytes) -> Optional[Dict[str, Any]]:
        """Read a normal (non-diagnostic) record."""
        second_byte = buf.read(1)
        if len(second_byte) < 1:
            return None

        score_cp = struct.unpack('<h', first_byte + second_byte)[0]

        if self.nn_type == "DNN":
            num_features = struct.unpack('<B', buf.read(1))[0]
            features = []
            for _ in range(num_features):
                features.append(struct.unpack('<H', buf.read(2))[0])

            return {
                'score_cp': score_cp,
                'features': features,
                'is_diagnostic': False
            }
        else:  # NNUE
            stm = struct.unpack('<B', buf.read(1))[0]

            num_white = struct.unpack('<B', buf.read(1))[0]
            white_features = []
            for _ in range(num_white):
                white_features.append(struct.unpack('<H', buf.read(2))[0])

            num_black = struct.unpack('<B', buf.read(1))[0]
            black_features = []
            for _ in range(num_black):
                black_features.append(struct.unpack('<H', buf.read(2))[0])

            return {
                'score_cp': score_cp,
                'stm': stm,
                'white_features': white_features,
                'black_features': black_features,
                'is_diagnostic': False
            }


def find_shards(base_dir: str = 'data', nn_type: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Find DNN and/or NNUE shard files in a directory.

    Args:
        base_dir: Base data directory
        nn_type: If specified, only find shards for this type ("DNN" or "NNUE")

    Returns:
        Tuple of (dnn_shards, nnue_shards) lists
    """
    dnn_shards = []
    nnue_shards = []

    if nn_type is None or nn_type.upper() == "DNN":
        patterns = [
            os.path.join(base_dir, 'dnn', '*.bin.zst'),
            os.path.join(base_dir, 'DNN', '*.bin.zst'),
        ]
        for pattern in patterns:
            dnn_shards.extend(glob.glob(pattern))

    if nn_type is None or nn_type.upper() == "NNUE":
        patterns = [
            os.path.join(base_dir, 'nnue', '*.bin.zst'),
            os.path.join(base_dir, 'NNUE', '*.bin.zst'),
        ]
        for pattern in patterns:
            nnue_shards.extend(glob.glob(pattern))

    return sorted(set(dnn_shards)), sorted(set(nnue_shards))


def discover_shards(data_dir: str, nn_type: str) -> List[str]:
    """
    Discover all shard files for a specific NN type.

    Args:
        data_dir: Data directory
        nn_type: "DNN" or "NNUE"

    Returns:
        List of shard file paths
    """
    pattern = os.path.join(data_dir, "*.bin.zst")
    shards = sorted(glob.glob(pattern))

    if not shards:
        # Try looking in nn_type subdirectory
        pattern = os.path.join(data_dir, nn_type.lower(), "*.bin.zst")
        shards = sorted(glob.glob(pattern))

    return shards