import { Flex } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { RuntimeWipNotice } from '../../../components/runtime-wip/runtime-wip';
import { useInferenceParams } from './use-inference-params';

export const InferencePage = () => {
    const { model_id } = useInferenceParams();

    const {
        data: { model },
    } = $api.useSuspenseQuery('get', '/api/models/{model_id}', {
        params: { path: { model_id } },
    });

    return (
        <Flex height='100%' justifyContent='center' alignItems='center'>
            <RuntimeWipNotice feature={`Running ${model.name}`} />
        </Flex>
    );
};
