from mytriton.trace import F32, I32, BlockType, PointerType, VectorType, make_block_type


def test_block_type_str_and_numel():
    ty = BlockType((16, 32), F32)

    assert ty.shape == (16, 32)
    assert ty.element == F32
    assert ty.numel == 512
    assert str(ty) == "block<16 x 32 x f32>"


def test_block_type_pointer_element():
    ty = BlockType((8, 16), PointerType(F32))

    assert ty.numel == 128
    assert str(ty) == "block<8 x 16 x ptr<f32>>"


def test_make_block_type_keeps_vector_type_for_1d():
    ty = make_block_type((256,), I32)

    assert ty == VectorType(256, I32)
    assert str(ty) == "vector<256 x i32>"
