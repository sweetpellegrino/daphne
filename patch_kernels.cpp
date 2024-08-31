// ----------------------------------------------------------------------------
// aggAll
// ----------------------------------------------------------------------------
void _sumAll__DenseMatrix_double__DenseMatrix_double(DenseMatrix<double> ** res, const DenseMatrix<double> * arg, int kId, DCTX(ctx)) {
    try{
        preKernelInstrumentation(kId, ctx);
        auto _m_res = DataObjectFactory::create<DenseMatrix<double>>(1,1,true);
        double _res = aggAll<double, DenseMatrix<double>>(AggOpCode::SUM, arg, ctx);
        _m_res ->set(0, 0, _res);
        *res = _m_res;
        postKernelInstrumentation(kId, ctx);
    } catch(std::exception &e) {
        throw ErrorHandler::runtimeError(kId, e.what(), &(ctx->dispatchMapping));
    }
}
void _sumAll__DenseMatrix_float__DenseMatrix_float(DenseMatrix<float> ** res, const DenseMatrix<float> * arg, int kId, DCTX(ctx)) {
    try{
        preKernelInstrumentation(kId, ctx);
        auto _m_res = DataObjectFactory::create<DenseMatrix<float>>(1,1,true);
        float _res = aggAll<float, DenseMatrix<float>>(AggOpCode::SUM, arg, ctx);
        _m_res ->set(0, 0, _res);
        *res = _m_res;
        postKernelInstrumentation(kId, ctx);
    } catch(std::exception &e) {
        throw ErrorHandler::runtimeError(kId, e.what(), &(ctx->dispatchMapping));
    }
}
void _sumAll__DenseMatrix_int64_t__DenseMatrix_int64_t(DenseMatrix<int64_t> ** res, const DenseMatrix<int64_t> * arg, int kId, DCTX(ctx)) {
    try{
        preKernelInstrumentation(kId, ctx);
        auto _m_res = DataObjectFactory::create<DenseMatrix<int64_t>>(1,1,true);
        int64_t _res = aggAll<int64_t, DenseMatrix<int64_t>>(AggOpCode::SUM, arg, ctx);
        _m_res ->set(0, 0, _res);
        *res = _m_res;
        postKernelInstrumentation(kId, ctx);
    } catch(std::exception &e) {
        throw ErrorHandler::runtimeError(kId, e.what(), &(ctx->dispatchMapping));
    }
}