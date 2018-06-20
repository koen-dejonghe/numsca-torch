package botkop.numsca

import botkop.{numsca => ns}
import org.scalatest.{FlatSpec, Matchers}
import torch.cpu.{SWIGTYPE_p_void, TH}

import scala.language.postfixOps

class TensorSpec extends FlatSpec with Matchers {

  "A tensor" should "create with provided data" in {

    val data = (1 until 7).toArray.map(_.toFloat)
    val shape = List(2, 3)
    val a = ns.create(data, shape)

    a.dim shouldBe 2
    a.shape shouldBe List(2, 3)
    a.size shouldBe 6
    a.data shouldBe Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
  }

  it should "make a zero array" in {
    val shape = List(2, 3)
    val z = ns.zeros(shape)
    z.shape shouldBe List(2, 3)
    z.data shouldBe Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  }

  /*
  it should "free when no longer used" in {
    val t3 = ns.ones(List(100, 100)) // this one will only get garbage collected at the end of the program
    for (_ <- 0 until 100) {
      ns.zeros(List(3000, 3000)) // these will get GC'ed as soon as as System.gc() is called
      Thread.sleep(1)
    }
    t3.desc shouldBe "torch.xTensor of size 100x100"
    t3.data.sum shouldBe 100 * 100
  }

  it should "free in parallel when no longer used" in {
    import scala.concurrent.ExecutionContext.Implicits.global
    import scala.concurrent.duration._

    val t3 = ns.ones(100, 100) // this one will only get garbage collected at the end of the program

    val futures = Future.sequence {
      (0 until 100).map { _ =>
        Future {
          ns.zeros(3000, 3000) // these will get GC'ed as soon as as System.gc() is called
          Thread.sleep(10)
        }
      }
    }

    Await.result(futures, 10 seconds)
    t3.desc shouldBe "torch.xTensor of size 100x100"
    t3.data.sum shouldBe 100 * 100
  }
  */

  it should "arange" in {
    val t = ns.arange(max = 10.0)
    t.shape shouldBe List(10)
    t.data shouldBe Array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

    val u = ns.arange(max = 10.0, step = 2.5)
    u.shape shouldBe List(4)
    u.data shouldBe Array(0.0, 2.5, 5.0, 7.5)
  }

  it should "seed" in {
    ns.setSeed(213L)
    val t = ns.randn(2, 3)
    println(t.data.toList)
    // hard to test
  }

  it should "randn" in {
    ns.setSeed(213L)
    val t = ns.randn(1000)
    t.shape shouldBe List(1000)
    val data = t.data
    val mean = data.sum / data.length
    println(mean)
    mean should be(0.0f +- 1e-1f)
    val std =
      Math.sqrt(data.map(d => Math.pow(d - mean, 2.0)).sum / (data.length - 1))
    println(std)
    std should be(1.0 +- 1e-1)
  }

  it should "randint" in {
    ns.setSeed(213L)
    val t = ns.randint(high = 10.0, shape = List(10))
    val data = t.data
    println(data.toList)
  }

  it should "linspace" in {
    val steps = 5
    val t = ns.linspace(0, 1, steps)
    t.shape shouldBe List(steps)
    val data = t.data
    data shouldBe Array(0.0, 0.25, 0.5, 0.75, 1.0)
  }

  //--------------------
  it should "cmul" in {
    val t1 = ns.arange(1, 10)
    val t2 = ns.arange(2, 11)
    val t3 = ns.mul(t2, t1)
    println(t3.shape)
    println(t3.data.toList)
  }

  it should "reshape" in {
    val a = ns.arange(max = 9)
    println(a.array.getStorage.getRefcount)
    val b = a.reshape(List(3, 3))
    println(a.array.getStorage.getRefcount)
    println(a.array.getStorage.getRefcount)

    b(1, 1) := 999

    a.shape shouldBe List(9)
    b.shape shouldBe List(3, 3)
    a isSameAs b shouldBe false
    a.data shouldBe b.data
    a.data shouldBe Array(0.0, 1.0, 2.0, 3.0, 999.0, 5.0, 6.0, 7.0, 8.0)

  }

  it should "select" in {
    val a = ns.arange(max = 9).reshape(3, 3)
    val b = ns.select(a, dimension = 0, sliceIndex = 1)
    println(b.shape)
    println(b.data.toList)

  }

  it should "select 2" in {
    val a = ns.arange(max = 8).reshape(2, 2, 2)
    // val r = NDArray.select(a, List(0, 1, 0))
    val r = a(0, 1, 0)

    println(r.shape)
    println(r.data.toList)
  }

  it should "assign to a selection" in {
    val a = ns.arange(max = 8).reshape(2, 2, 2)
    val r = ns.randn(1)
    a(0, 1, 0) := r
    println(a.data.toList)

    val r2 = ns.fill(3.14f, List(2, 2))
    a(1) := r2
    println(a.data.toList)

    // broadcasting
    val r3 = ns.tensor(100)
    a(0) := r3
    println(a.data.toList)
  }

  it should "narrow" in {
    val a = ns.arange(max = 8).reshape(2, 2, 2)
    val b = ns.narrow(a, dimension = 0, firstIndex = 1, size = 1)
    ns.setValue(b, 3.17f, List(0, 0, 0))
    println(b.shape)
    println(b.data.toList)
    println(a.data.toList)
  }

  it should "linear" in {
    val x = ns.randint(1, 5, List(2, 3))
    val y = ns.randint(1, 5, List(3, 4))
    val b = ns.randint(1, 5, List(2, 4))

    val r = ns.linear(x, y, b)
    println(r.shape)
    println(r.data.toList)

    println(r(0, 1))
  }

  //============================
  // numsca tests

  val ta: Tensor = ns.arange(max = 10)
  val tb: Tensor = ns.reshape(ns.arange(max = 9), 3, 3)
  val tc: Tensor = ns.reshape(ns.arange(max = 2 * 3 * 4), 2, 3, 4)

  // Elements
  it should "retrieve the correct elements" in {
    assert(ta(1).squeeze().data.head == 1)
    assert(tb(1, 0).squeeze().data.head == 3)
    assert(tc(1, 0, 2).squeeze().data.head == 14)

    val i = List(1, 0, 1)
    assert(tc(i: _*).squeeze().data.head == 13)
  }

  it should "change array values in place" in {
    val t = ta.copy()
    t(3) := -5f
    assert(t.data sameElements Array(0, 1, 2, -5, 4, 5, 6, 7, 8, 9))
    t(0) += 7
    println(t.data.toList)
    assert(t.data sameElements Array(7, 1, 2, -5, 4, 5, 6, 7, 8, 9))

    val t2 = tb.copy()
    t2(2, 1) := -7
    t2(1, 2) := -3
    /*
    assert(
      arrayEqual(t2,
        Tensor(0.00, 1.00, 2.00, 3.00, 4.00, -3.00, 6.00, -7.00,
          8.00).reshape(3, 3)))
     */
    assert(t2.shape == List(3, 3))
    assert(
      t2.data.toList == List(0.0, 1.0, 2.0, 3.0, 4.0, -3.0, 6.0, -7.0, 8.0))
  }

  it should "do operations array-wise" in {
    val a2 = ta * 2
    assert(a2.data sameElements Array(0, 2, 4, 6, 8, 10, 12, 14, 16, 18))
  }

  it should "slice over a single dimension" in {

    val a0 = ns.arange(max = 10)

    // A[1:]
    val a1 = a0(1 :>)

    // A[:-1]
    val a2 = a0(0 :> -1)

    // A[1:] - A[:-1]
    val a3 = a1 - a2
    println(a1)
    println(a2)
    println(a3)

    assert(ns.equal(a3, ns.fill(1.0f, List(9))))

    assert(a0(5 :>) isSameAs ns.tensor(5, 6, 7, 8, 9))
    assert(a0(0 :> 5) isSameAs ns.tensor(0, 1, 2, 3, 4))

    val s = 3 :> -1
    ta(s) isSameAs ns.tensor(3, 4, 5, 6, 7, 8) shouldBe true

  }

  it should "update over a single dimension" in {

    val t = ns.arange(max = 10)
    t(2 :> 5) := -ns.ones(3)
    assert(t isSameAs ns.tensor(0, 1, -1, -1, -1, 5, 6, 7, 8, 9))

    /* does not work: crashes instead
    an[Throwable] should be thrownBy {
      t(2 :> 5) := -ns.ones(4)
    }
     */

    t(2 :> 5) := 33
    t isSameAs ns.tensor(0, 1, 33, 33, 33, 5, 6, 7, 8, 9) shouldBe true

    t(2 :> 5) -= 1
    t isSameAs ns.tensor(0, 1, 32, 32, 32, 5, 6, 7, 8, 9) shouldBe true

    t := -1
    t isSameAs ns.fill(-1, List(10)) shouldBe true
  }

  it should "slice over multiple dimensions" in {
    val tb = ns.arange(max = 9).reshape(3, 3)
    val b1 = tb(0 :> 2, :>)
    b1 isSameAs ns.arange(max = 6).reshape(2, 3) shouldBe true
  }

  it should "slice over multiple dimensions with integer indexing" in {
    val b2 = tb(1, 0 :> -1)
    b2 isSameAs ns.tensor(3, 4) shouldBe true
  }

  it should "add" in {
    val x = ns.arange(max = 4)
    val y = ns.ones(4)
    val z = x + y

    x isSameAs ns.create(0, 1, 2, 3) shouldBe true
    y isSameAs ns.create(1, 1, 1, 1) shouldBe true
    z isSameAs ns.create(1, 2, 3, 4) shouldBe true
  }

  it should "broadcast operations" in {
    val x = ns.arange(max = 4)
    val xx = x.reshape(4, 1)
    val y = ns.ones(5)
    val z = ns.ones(3, 4)

    // does not work
//    try {
//      println(x + y)
//    } catch {
//      case t: Throwable =>
//        println("caught")
//    }

    (xx + y).shape shouldBe List(4, 5)

    val s1 = ns.tensor(
      1, 1, 1, 1, 1, //
      2, 2, 2, 2, 2, //
      3, 3, 3, 3, 3, //
      4, 4, 4, 4, 4 //
    ).reshape(4, 5)
    (xx + y) isSameAs s1 shouldBe true

    val s2 = ns.tensor(
      1, 2, 3, 4, //
      1, 2, 3, 4, //
      1, 2, 3, 4 //
    ).reshape(3, 4)
    (x + z) isSameAs s2 shouldBe true

    // also test same shape
    val t1 = ns.arange(max = 9).reshape(3, 3)
    val t2 = t1 + t1
    assert(t2 isSameAs t1 * 2)

  }

  it should "outer sum" in {
    // outer sum
    val a = ns.tensor(0, 10, 20, 30).reshape(4, 1)
    val b = ns.tensor(1, 2, 3)
    val c = ns.tensor(
      1, 2, 3, //
      11, 12, 13, //
      21, 22, 23, //
      31, 32, 33 //
    ).reshape(4, 3)

    println(c)

    (a + b) isSameAs c shouldBe true

    val observation = ns.tensor(111, 188)
    val codes = ns.tensor(
      102, 203, //
      132, 193, //
      45, 155, //
      57, 173 //
    ).reshape(4, 2)
    val diff = codes - observation

    println(diff)
    val sq = ns.square(diff)
    println(sq)
    val sum = ns.sum(sq, axis = -1)
    println(sum)
    val am = ns.argmin(sum, 0).squeeze()
    assert(am.value(0) == 0)
  }

  it should "expand nd" in {

    def verify(shape1: List[Int],
               shape2: List[Int],
               expectedShape: List[Int]): Unit = {

      val a1 = ns.ones(shape1)
      val a2 = ns.ones(shape2)
      val Seq(a3, a4) = ns.expandNd(Seq(a2, a1))
      println(a3.shape)
      println(a4.shape)
      a3.shape shouldBe expectedShape
      a4.shape shouldBe expectedShape

      println(a3)
      println(a4)

//      a3 isSameAs ns.zeros(expectedShape) shouldBe true
//      a4 isSameAs ns.ones(expectedShape) shouldBe true

      ns.sum(a3) shouldBe ns.sum(a4)

    }

    verify(List(5, 4), List(1), List(5, 4))
    verify(List(256, 256, 3), List(3), List(256, 256, 3))
    verify(List(15, 3, 5), List(15, 1, 5), List(15, 3, 5))
    verify(List(15, 3, 5), List(3, 5), List(15, 3, 5))
    verify(List(15, 3, 5), List(3, 1), List(15, 3, 5))
    // bidirectional
    verify(List(8, 1, 6, 1), List(7, 1, 5), List(8, 7, 6, 5))

  }

  it should "index select" in {
    val primes = ns.tensor(2, 3, 5, 7, 11, 13, 17, 19, 23)
    val idx = ns.tensor(3, 4, 1, 2, 2)
    val r = ns.indexSelect(primes, 0, idx)
    r isSameAs ns.tensor(7, 11, 3, 5, 5) shouldBe true

    println(primes)
    println(r(1))
    // r(1) := 0f

    TH.THFloatTensor_set1d(r.array, 1, 377f)

    println(primes)
    println(r)

    /*
    val numSamples = 4
    val numClasses = 2
    val x = ns.arange(min = 10, max = numSamples * numClasses + 10).reshape(numSamples, numClasses)
    println(x)
    val y = NDArray(1, 0, 3, 2)
    val s = ns.indexSelect(x, 0, y)
    println(s)
    */

  }

  it should "list-of-location index" in {
    val numSamples = 4
    val numClasses = 3
    val x = ns.arange(max = numSamples * numClasses).reshape(numSamples, numClasses)

    val y = ns.tensor(0, 1, 2, 1)
    val z = x.select(ns.arange(max = numSamples), y).squeeze()

    println(z)
  }

  it should "ix select" in {
    val primes = ns.tensor(2, 3, 5, 7, 11, 13, 17, 19, 23)
    val idx = List(3, 4, 1, 2, 2)
    val r = ns.ixSelect(primes, idx)

    println(r)

    // r isSameAs ns.tensor(7, 11, 3, 5, 5) shouldBe true

    println(primes)
    println(r(1))
    r(1) := 9999

    println(primes)
    println(r)

  }

  it should "simple nn" in {

    val x = ns.arange(max = 12).reshape(List(4, 3))
    val w = ns.arange(max = 80).reshape(20, 3)
    val b = ns.arange(max = 20)

    val state: SWIGTYPE_p_void = null

    val y = ns.empty
    val buffer = ns.empty


    // public static void THNN_FloatLinear_updateOutput(SWIGTYPE_p_void state, THFloatTensor input, THFloatTensor output, THFloatTensor weight, THFloatTensor bias, THFloatTensor addBuffer) {
    TH.THNN_FloatLinear_updateOutput(state, x.array, y.array, w.array, b.array, buffer.array)

    println(y)
    println
    println(buffer)
  }

}
