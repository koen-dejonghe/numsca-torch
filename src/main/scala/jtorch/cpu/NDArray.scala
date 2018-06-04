package jtorch.cpu

import java.math.BigInteger

import botkop.numsca.MemoryManager
import com.typesafe.scalalogging.LazyLogging

import scala.language.postfixOps

class NDArray private (val payload: SWIGTYPE_p_THFloatTensor) extends LazyLogging {

  def dim: Int = TH.THFloatTensor_nDimension(payload)
  def size(dim: Int): Int = TH.THFloatTensor_size(payload, dim).toInt
  def shape: List[Int] = (0 until dim) map size toList
  def size: Int = shape.product
  def desc: String = TH.THFloatTensor_desc(payload).getStr
  def numel: Int = TH.THFloatTensor_numel(payload)

  def data: Array[Float] = {
    val p = TH.THFloatTensor_data(payload)
    (0 until size).map(TH.floatArray_getitem(p, _)) toArray
  }

  /**
    * Free tensor and notify memory manager
    */
  override def finalize(): Unit = {
    val memSize = MemoryManager.dec(size) // obtain size before freeing!
    logger.debug(s"freeing (mem = $memSize)")
    TH.THFloatTensor_free(payload)
  }

}

object NDArray extends LazyLogging {

  val rng: SWIGTYPE_p_THGenerator = TH.THGenerator_new()

//  def initialSeed: Long = TH.THRandom_initialSeed(rng).longValue()
//  def currentSeed: Long = TH.THRandom_seed(rng).longValue()
  def setSeed(theSeed: Long): Unit = TH.THRandom_manualSeed(rng, BigInteger.valueOf(theSeed))

  def create(data: Array[Float], shape: List[Int]): NDArray = {
    require(data.length == shape.product)
    MemoryManager.memCheck(shape)
    val size = data.length
    val a = floatArray(data)
    val storage: SWIGTYPE_p_THFloatStorage =
      TH.THFloatStorage_newWithData(a, size)
    val t =
      TH.THFloatTensor_newWithStorage(storage, 0, longStorage(shape), null)
    new NDArray(t)
  }

  def create(data: Array[Float]): NDArray = create(data, List(data.length))

  def create(data: Float*): NDArray = create(data.toArray)

  def zeros(shape: List[Int]): NDArray = {
    MemoryManager.memCheck(shape)
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_zeros(t, ls)
    new NDArray(t)
  }

  def zeros(shape: Int*): NDArray = zeros(shape.toList)

  def ones(shape: List[Int]): NDArray = {
    MemoryManager.memCheck(shape)
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_ones(t, ls)
    new NDArray(t)
  }

  def ones(shape: Int*): NDArray = ones(shape.toList)

  def fill(f: Float, shape: List[Int]): NDArray = {
    MemoryManager.memCheck(shape)
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_newWithSize(ls, null)
    TH.THFloatTensor_fill(t, f)
    new NDArray(t)
  }

  def arange(min: Double = 0, max: Double, step: Double = 1): NDArray = {
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_arange(t, min, max, step)
    new NDArray(t)
  }

  def randn(shape: List[Int]): NDArray = {
    MemoryManager.memCheck(shape)
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_randn(t, rng, ls)
    new NDArray(t)
  }

  def randn(shape: Int*): NDArray = randn(shape.toList)

  def randint(low: Double = 0.0, high: Double, shape: List[Int]): NDArray = {
    MemoryManager.memCheck(shape)
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_newWithSize(ls, null)
    TH.THFloatTensor_uniform(t, rng, low, high)
    TH.THFloatTensor_floor(t, t)
    new NDArray(t)
  }

  def uniform(low: Double = 0.0, high: Double, shape: List[Int]): NDArray = {
    MemoryManager.memCheck(shape)
    val ls = longStorage(shape)
    val t = TH.THFloatTensor_newWithSize(ls, null)
    TH.THFloatTensor_uniform(t, rng, low, high)
    new NDArray(t)
  }

  def linspace(start: Float, end: Float, steps: Long): NDArray = {
    MemoryManager.memCheck(steps)
    val t = TH.THFloatTensor_new
    TH.THFloatTensor_linspace(t, start, end, steps)
    new NDArray(t)
  }

  // operations that create a new tensor -----------------------------
  def cmul(t1: NDArray, t2: NDArray): NDArray = {
    val r = TH.THFloatTensor_new
    TH.THFloatTensor_cmul(r, t1.payload, t2.payload)
    MemoryManager.memCheck(TH.THFloatTensor_numel(r))
    new NDArray(r)
  }

  // utility functions -----------------------------

  private def longStorage(shape: Seq[Int]): SWIGTYPE_p_THLongStorage = {
    val size = shape.length
    val data = TH.new_longLongArray(size)
    var i = 0
    while (i < shape.length) {
      TH.longLongArray_setitem(data, i, shape(i))
      i = i + 1
    }
    TH.THLongStorage_newWithData(data, size)
  }

  private def floatArray(data: Array[Float]): SWIGTYPE_p_float = {
    val size = data.length
    val a: SWIGTYPE_p_float = TH.new_floatArray(size)
    var i = 0
    while (i < size) {
      TH.floatArray_setitem(a, i, data(i))
      i = i + 1
    }
    a
  }

}
