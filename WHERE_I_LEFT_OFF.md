Things were going pretty well. This should have been easy.

Making the model seemingly goes well. After it's made tensorflow serving runs it but all inference against it yields weird errors like:
```
You must feed a value for placeholder tensor 'decode_length_lol' with dtype int32
```

And that's after feeding it an int32.

Trying to get the model going locally fails too. This won't even work:

```python
dirs = [x for x in os.listdir('/Users/joseph.weidinger/git/personal/magenta/export')]
latest_dir = dirs[-1]
thing = tf.saved_model.load(f"/Users/joseph.weidinger/git/personal/magenta/export/{latest_dir}", tags=["serve"])
f = thing.signatures["serving_default"]
```

This yields the error:
```
Unable to lift tensor <tf.Tensor 'transformer/strided_slice_1:0' shape=(None, None) dtype=int64> because it depends transitively on placeholder <tf.Operation 'targets_lol' type=Placeholder> via at least one path, e.g.: transformer/strided_slice_1 (StridedSlice) <- transformer/strided_slice_1/stack (Pack) <- transformer/strided_slice (StridedSlice) <- transformer/Shape (Shape) <- transformer/ToInt64 (Cast) <- targets_lol (Placeholder)
```

At this point I can't really make heads or tails about what could be going on.

Oh ya, and one more thing... Even though for non-SavedModel inference, you only seemingly need `decode_length` and `targets` but the tensorflow serving server complains unless you give it `input_space_id` and `target_space_id`, which are both in32 scalars I think. Not even sure what they mean though. When I run the debugger through a non-SavedModel eval, you can see it making those, but I don't know what they mean.

