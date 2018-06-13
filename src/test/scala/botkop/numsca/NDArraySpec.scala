package botkop.numsca

import org.scalatest.{FlatSpec, Matchers}

import scala.concurrent.{Await, Future}
import scala.language.postfixOps
import botkop.numsca.{NDArray => nd}
import torch.cpu.{TH, THJNI}

class NDArraySpec extends FlatSpec with Matchers {

  "An ND array" should "create with provided data" in {

    val data = (1 until 7).toArray.map(_.toFloat)
    val shape = List(2, 3)
    val a = NDArray.create(data, shape)

    a.dim shouldBe 2
    a.shape shouldBe List(2, 3)
    a.size shouldBe 6
    a.data shouldBe Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
  }

  it should "make a zero array" in {
    val shape = List(2, 3)
    val z = NDArray.zeros(shape)
    z.shape shouldBe List(2, 3)
    z.data shouldBe Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  }

  it should "free when no longer used" in {
    val t3 = NDArray.ones(List(100, 100)) // this one will only get garbage collected at the end of the program
    for (_ <- 0 until 100) {
      NDArray.zeros(List(3000, 3000)) // these will get GC'ed as soon as as System.gc() is called
      Thread.sleep(1)
    }
    t3.desc shouldBe "torch.xTensor of size 100x100"
    t3.data.sum shouldBe 100 * 100
  }

  it should "free in parallel when no longer used" in {
    import scala.concurrent.ExecutionContext.Implicits.global
    import scala.concurrent.duration._

    val t3 = NDArray.ones(100, 100) // this one will only get garbage collected at the end of the program

    val futures = Future.sequence {
      (0 until 100).map { _ =>
        Future {
          NDArray.zeros(3000, 3000) // these will get GC'ed as soon as as System.gc() is called
          Thread.sleep(10)
        }
      }
    }

    Await.result(futures, 10 seconds)
    t3.desc shouldBe "torch.xTensor of size 100x100"
    t3.data.sum shouldBe 100 * 100
  }

  it should "arange" in {
    val t = NDArray.arange(max = 10.0)
    t.shape shouldBe List(10)
    t.data shouldBe Array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

    val u = NDArray.arange(max = 10.0, step = 2.5)
    u.shape shouldBe List(4)
    u.data shouldBe Array(0.0, 2.5, 5.0, 7.5)
  }

  it should "seed" in {
    NDArray.setSeed(213L)
    val t = NDArray.randn(2, 3)
    println(t.data.toList)
    // hard to test
  }

  it should "randn" in {
    NDArray.setSeed(213L)
    val t = NDArray.randn(1000)
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
    NDArray.setSeed(213L)
    val t = NDArray.randint(high = 10.0, shape = List(10))
    val data = t.data
    println(data.toList)
  }

  it should "linspace" in {
    val steps = 5
    val t = NDArray.linspace(0, 1, steps)
    t.shape shouldBe List(steps)
    val data = t.data
    data shouldBe Array(0.0, 0.25, 0.5, 0.75, 1.0)
  }

  //--------------------
  it should "cmul" in {
    val t1 = NDArray.arange(1, 10)
    val t2 = NDArray.arange(2, 11)
    val t3 = NDArray.mul(t2, t1)
    println(t3.shape)
    println(t3.data.toList)
  }

  it should "reshape" in {
    val a = NDArray.arange(max = 9)
    println(a.payload.getStorage.getRefcount)
    val b = a.reshape(List(3, 3))
    println(a.payload.getStorage.getRefcount)
    println(a.payload.getStorage.getRefcount)

    b(1, 1) := 999

    a.shape shouldBe List(9)
    b.shape shouldBe List(3, 3)
    a isSameAs b shouldBe false
    a.data shouldBe b.data
    a.data shouldBe Array(0.0, 1.0, 2.0, 3.0, 999.0, 5.0, 6.0, 7.0, 8.0)
    a.payload.getStorage.getRefcount shouldBe 2

  }

  it should "select" in {
    val a = NDArray.arange(max = 9).reshape(3, 3)
    val b = NDArray.select(a, dimension = 0, sliceIndex = 1)
    println(b.shape)
    println(b.data.toList)

  }

  it should "select 2" in {
    val a = NDArray.arange(max = 8).reshape(2, 2, 2)
    // val r = NDArray.select(a, List(0, 1, 0))
    val r = a(0, 1, 0)

    println(r.shape)
    println(r.data.toList)
  }

  it should "assign to a selection" in {
    val a = NDArray.arange(max = 8).reshape(2, 2, 2)
    val r = NDArray.randn(1)
    a(0, 1, 0) := r
    println(a.data.toList)

    val r2 = NDArray.fill(3.14f, List(2, 2))
    a(1) := r2
    println(a.data.toList)

    // broadcasting
    val r3 = NDArray(100)
    a(0) := r3
    println(a.data.toList)
  }

  it should "narrow" in {
    val a = NDArray.arange(max = 8).reshape(2, 2, 2)
    val b = NDArray.narrow(a, dimension = 0, firstIndex = 1, size = 1)
    NDArray.setValue(b, 3.17f, List(0, 0, 0))
    println(b.shape)
    println(b.data.toList)
    println(a.data.toList)
  }

  it should "linear" in {
    val x = NDArray.randint(1, 5, List(2, 3))
    val y = NDArray.randint(1, 5, List(3, 4))
    val b = NDArray.randint(1, 5, List(2, 4))

    val r = NDArray.linear(x, y, b)
    println(r.shape)
    println(r.data.toList)

    println(r(0, 1))
  }

  //============================
  // numsca tests

  val ta: NDArray = NDArray.arange(max = 10)
  val tb: NDArray = NDArray.reshape(NDArray.arange(max = 9), 3, 3)
  val tc: NDArray = NDArray.reshape(NDArray.arange(max = 2 * 3 * 4), 2, 3, 4)

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

    val a0 = NDArray.arange(max = 10)

    // A[1:]
    val a1 = a0(1 :>)

    // A[:-1]
    val a2 = a0(0 :> -1)

    // A[1:] - A[:-1]
    val a3 = a1 - a2
    println(a1)
    println(a2)
    println(a3)

    assert(NDArray.equal(a3, NDArray.fill(1.0f, List(9))))

    assert(a0(5 :>) isSameAs NDArray(5, 6, 7, 8, 9))
    assert(a0(0 :> 5) isSameAs NDArray(0, 1, 2, 3, 4))

    val s = 3 :> -1
    ta(s) isSameAs nd(3, 4, 5, 6, 7, 8) shouldBe true

  }

  it should "update over a single dimension" in {

    val t = nd.arange(max = 10)
    t(2 :> 5) := -nd.ones(3)
    assert(t isSameAs nd(0, 1, -1, -1, -1, 5, 6, 7, 8, 9))

    /* does not work: crashes instead
    an[Throwable] should be thrownBy {
      t(2 :> 5) := -nd.ones(4)
    }
     */

    t(2 :> 5) := 33
    t isSameAs nd(0, 1, 33, 33, 33, 5, 6, 7, 8, 9) shouldBe true

    t(2 :> 5) -= 1
    t isSameAs nd(0, 1, 32, 32, 32, 5, 6, 7, 8, 9) shouldBe true

    t := -1
    t isSameAs nd.fill(-1, List(10)) shouldBe true
  }

  it should "slice over multiple dimensions" in {
    val tb = nd.arange(max = 9).reshape(3, 3)
    val b1 = tb(0 :> 2, :>)
    b1 isSameAs nd.arange(max = 6).reshape(2, 3) shouldBe true
  }

  it should "slice over multiple dimensions with integer indexing" in {
    val b2 = tb(1, 0 :> -1)
    b2 isSameAs nd(3, 4) shouldBe true
  }

  it should "broadcast with another tensor" in {

    def verify(shape1: List[Int],
               shape2: List[Int],
               expectedShape: List[Int]): Unit = {

      val a1 = nd.zeros(shape1)
      val a2 = nd.zeros(shape2)
      val a3 = nd.expand(a2, a1)
      a3.shape shouldBe expectedShape
    }

    verify(List(256, 256, 3), List(3), List(256, 256, 3))
    verify(List(5, 4), List(1), List(5, 4))
    verify(List(15, 3, 5), List(15, 1, 5), List(15, 3, 5))
    verify(List(15, 3, 5), List(3, 5), List(15, 3, 5))
    verify(List(15, 3, 5), List(3, 1), List(15, 3, 5))

  }

  it should "broadcast in both directions" in {

    def verify(shape1: List[Int],
               shape2: List[Int],
               expectedShape: List[Int]): Unit = {

      val a1 = nd.zeros(shape1)
      val a2 = nd.zeros(shape2)
      val Seq(a3, a4) = nd.expand(Seq(a2, a1))
      println(a3.shape)
      println(a4.shape)
      a3.shape shouldBe expectedShape
      a4.shape shouldBe expectedShape
    }

    // bidirectional
    verify(List(8, 1, 6, 1), List(7, 1, 5), List(8, 7, 6, 5))
  }

  it should "broadcast operations" in {
    val x = nd.arange(max = 4)
    val xx = x.reshape(4, 1)
    val y = nd.ones(5)
    val z = nd.ones(3, 4)

    // does not work
//    try {
//      println(x + y)
//    } catch {
//      case t: Throwable =>
//        println("caught")
//    }

    (xx + y).shape shouldBe List(4, 5)

    val s1 = nd(
      1, 1, 1, 1, 1, //
      2, 2, 2, 2, 2, //
      3, 3, 3, 3, 3, //
      4, 4, 4, 4, 4 //
    ).reshape(4, 5)
    (xx + y) isSameAs s1 shouldBe true

    val s2 = nd(
        1, 2, 3, 4, //
        1, 2, 3, 4, //
        1, 2, 3, 4 //
      ).reshape(3, 4)
    (x + z) isSameAs s2 shouldBe true

    // outer sum
    val a = nd(0, 10, 20, 30).reshape(4, 1)
    val b = nd(1, 2, 3)
    val c = nd(
      1, 2, 3, //
      11, 12, 13, //
      21, 22, 23, //
      31, 32, 33 //
    ).reshape(4, 3)

    (a + b) isSameAs c shouldBe true

    val observation = nd(111, 188)
    val codes = nd(
      102, 203, //
      132, 193, //
      45, 155, //
      57, 173 //
    ).reshape(4, 2)
    val diff = codes - observation
//    val dist = nd.sqrt(nd.sum(nd.square(diff), axis = -1))
//    val nearest = nd.argmin(dist).squeeze()
//    assert(nearest == 0)
  }

  it should "aaa" in {
    val a = nd.arange(max = 10)
    println(a.payload.getStorage.getRefcount)
    println(a.payload.getStorage.getData)
    val b = a.reshape(5, 2)
    println(b.payload.getStorage.getRefcount)
    println(a.payload.getStorage.getData)

    b(3) := 9999

    println(a)
    println(b)

  }

}
